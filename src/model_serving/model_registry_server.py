# cli run `redis-server --daemonize yes` to start the redis server

from flask import Flask, request, jsonify
from requests.exceptions import ConnectionError
import requests
import argparse
import random
import logging
import redis

from typing import List, Dict

import sys
from http_utils import test_worker

app = Flask(__name__)
logger = logging.getLogger(__name__)


class WorkerManager:
    def __init__(self, redis_host="localhost", redis_port=6379):
        self.redis_client = redis.StrictRedis(
            host=redis_host, port=redis_port, decode_responses=True
        )

    def setup(self, reset: bool = False):
        if reset:
            self.redis_client.flushdb()
        else:
            self.recover_worker_info()

    # def add_worker(self, worker_url: str, model_name: str) -> bool:
    #     if not self.redis_client.sismember(model_name, worker_url):
    #         self.redis_client.sadd(model_name, worker_url)
    #         return True
    #     return False

    def remove_worker(self, worker_url: str, model_name: str) -> bool:
        if self.redis_client.sismember(model_name, worker_url):
            self.redis_client.srem(model_name, worker_url)
            if self.redis_client.scard(model_name) == 0:
                self.redis_client.delete(model_name)
            return True
        return False

    # def get_worker(self, model_name: str) -> str:
    #     workers = list(self.redis_client.smembers(model_name))  # type: ignore
    #     if workers:
    #         return random.choice(workers)
    #     raise ValueError(f"No workers available for model {model_name}")

    # def get_all_workers(self, model_name: str, test_workers: bool = True) -> List[str]:
    #     if test_workers:
    #         self.test_model_workers(model_name)
    #     workers = list(self.redis_client.smembers(model_name))  # type: ignore
    #     if workers:
    #         return workers
    #     raise ValueError(f"Invalid model name {model_name}")

    def test_model_workers(self, model_name: str):
        workers = list(self.redis_client.smembers(model_name))  # type: ignore
        for url in workers:
            if not test_worker(url=url, model_name=model_name, interval=5, max_tries=2):
                print(
                    f"Removing unresponsive server {url} for {model_name}", flush=True
                )
                self.remove_worker(url, model_name)

    def print_all_workers(self):
        model_names = self.redis_client.keys()
        for model_name in model_names:  # type: ignore
            workers = list(self.redis_client.smembers(model_name))  # type: ignore
            print(f"{model_name}:", flush=True)
            for url in workers:
                print(f"\t* {url}", flush=True)

    def recover_worker_info(self):
        print("Recovered worker info (availability not tested):")
        self.print_all_workers()


# this will be init in the main function
worker_manager = WorkerManager()
# worker_manager = WorkerManager("worker_info.jsonl")


@app.route("/workers/list", methods=["POST"])
def list_worker():
    if (request_json := request.json) is not None:
        model_name = request_json.get("model_name")
    else:
        return jsonify({"error": "Data must be in JSON format!"}), 400

    if len(model_name.strip()) == 0:
        return (
            jsonify({"error": "Empty model_name!"}),
            400,
        )

    if model_name in worker_manager.redis_client.keys():
        serving_urls = worker_manager.get_all_workers(model_name)
        return jsonify({"workers": serving_urls}), 200
    else:
        return jsonify({"error": "Invalid model name"}), 400


@app.route("/workers/add", methods=["POST"])
def add_worker():
    if (request_json := request.json) is not None:
        worker_url = request_json.get("url")
        worker_model_name = request_json.get("model_name")
    else:
        return jsonify({"error": "Data must be in JSON format!"}), 400

    if len(worker_url.strip()) == 0 or len(worker_model_name.strip()) == 0:
        return (
            jsonify({"error": "Empty url or worker_name!"}),
            400,
        )

    if test_worker(worker_url, worker_model_name, interval=5, max_tries=2):
        if worker_manager.add_worker(worker_url, worker_model_name):
            return jsonify({"status": "Worker added"}), 200
        return jsonify({"error": "Failed to add worker"}), 400
    else:
        return (
            jsonify({"error": "Maximum tries elapses before the worker is online"}),
            400,
        )


@app.route("/workers/delete", methods=["POST"])
def delete_worker():
    if (request_json := request.json) is not None:
        worker_url = request_json.get("url")
        worker_model_name = request_json.get("model_name")
    else:
        return jsonify({"error": "Data must be in JSON format!"}), 400

    if worker_manager.remove_worker(worker_url, worker_model_name):
        return jsonify({"status": "Worker removed"}), 200
    return jsonify({"error": "Failed to remove worker"}), 400


@app.route("/<path:path>", methods=["GET", "POST", "PUT", "DELETE"])
def load_balancer(path):
    if (request_json := request.json) is not None:
        model_name = request_json.get("model")
    else:
        return jsonify({"error": "Data must be in JSON format!"}), 400

    try:
        worker = worker_manager.get_worker(model_name)
    except ValueError as e:
        return jsonify({"error": str(e)}), 503
    url = f"{worker}/{path}"
    # logger.info(f"## Dispatching {model_name} to {url}")

    try:
        if request.method == "POST":
            response = requests.post(url, json=request.json, headers=request.headers)
        else:
            return jsonify({"error": "Method not supported"}), 405

        # Filter out unwanted headers
        filtered_headers = {
            key: value
            for key, value in response.headers.items()
            if key.lower() != "transfer-encoding"
        }
        return response.content, response.status_code, filtered_headers
    except ConnectionError as e:
        logger.error(f"Error contacting {url}: {e}")
        # Remove the non-responsive worker
        worker_manager.remove_worker(worker, model_name)
        logger.info(
            f"Removed non-responsive worker {worker}. Retrying with another worker..."
        )
        # Recursive call to try the next available worker
        return load_balancer(path)


# app.run(host="0.0.0.0", port=5000, threaded=True)


def main():
    global worker_manager

    # add the args
    parser = argparse.ArgumentParser(description="Args for the registry server")
    parser.add_argument("--port", "-p", type=int, help="port", default=5000)
    parser.add_argument(
        "--model_name", "-m", type=str, help="model name to test the workers", default="meta-llama/Llama-3.1-405B-Instruct-FP8"
    )
    parser.add_argument(
        "--reset", action="store_true", help="whether to overwrite the file and reset"
    )
    parser.add_argument(
        "--save_file",
        "-f",
        type=str,
        help="file to save the worker info",
        default="worker_info.jsonl",
    )
    args = parser.parse_args()

    # init the worker_manager in the global context
    worker_manager.setup(reset=args.reset)
    print("Worker manager setup complete")

    # worker_manager.test_model_workers(model_name=args.model_name)
    # print("Worker manager test complete")

    print("Starting the server...")
    app.run(host="0.0.0.0", port=args.port, threaded=True)
    

if __name__ == "__main__":
    main()
