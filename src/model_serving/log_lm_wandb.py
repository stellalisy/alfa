import re
import time
import wandb
import requests
import argparse
import getpass

from typing import List, Tuple


def log_vllm_stats(url: str):
    try:
        # obtain the vllm stats
        response = requests.get(url)
        text_lines = response.text.split("\n")
        stats_lines = list(filter(lambda x: not x.startswith("#"), text_lines))

        metrics = {}
        for line in stats_lines:
            try:
                metric_val = float(line.split(" ")[-1])
            except ValueError:
                continue

            metric_name = line.split(" ")[0].replace("vllm:", "")
            metric_name = re.sub(r'model_name="[^"]*"', "", metric_name)

            metrics[metric_name] = metric_val

        wandb.log(metrics)

    except requests.exceptions.RequestException as e:
        print(f"Error sending HTTP request: {e}")
        return None


def main():
    # build the args
    parser = argparse.ArgumentParser(description="Args for the wandb logger.")
    parser.add_argument("--url", type=str, help="vllm url", default="http://localhost")
    parser.add_argument("--port", type=str, help="vllm port")
    parser.add_argument("--model_name", type=str, help="the model being served")
    parser.add_argument("--job_id", type=str, help="the slurm job id")
    parser.add_argument("--hostname", type=str, help="the hostname of the job")
    parser.add_argument("--global_rank", type=int, help="the global rank of this task")
    parser.add_argument(
        "--query_every_s", type=int, help="collect the stats every x seconds"
    )
    args = parser.parse_args()

    # start a new wandb run to track this script
    wandb.init(
        # set the wandb project where this run will be logged
        project="ts-model-serving",
        entity=getpass.getuser(),
        name=f"serving-{args.job_id}-rank{args.global_rank}",
        config={
            "model_name": args.model_name,
            "hostname": args.hostname,
            "job_id": args.job_id,
        },
        tags=["serving"],
    )

    # get the url from the args
    url = f"{args.url}:{args.port}/metrics"

    while True:
        log_vllm_stats(url)
        time.sleep(args.query_every_s)


if __name__ == "__main__":
    main()
