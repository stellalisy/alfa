import asyncio
import aiohttp
import requests
import logging
import time
import resource

from typing import List, Dict, Any, Tuple, Final, NamedTuple

logger = logging.getLogger(__name__)

# set some global timeout defaults (all in seconds)
# NOTE: `Final` is only for static checkers and not enforced at runtime
DEFAULT_REQUEST_TIMEOUT: Final = 600

CHAT_ENDPOINT: str = "/v1/chat/completions"
EXEC_ENDPOINT: str = "/exec"
COMP_ENDPOINT: str = "/v1/completions"

EMPTY_USAGE_DICT = {
    "usage": {
        "prompt_tokens": 0,
        "total_tokens": 0,
        "completion_tokens": 0,
    }
}

# Default number of file descriptors (including http connection) for python is 8,192
soft_limit, hard_limit = resource.getrlimit(resource.RLIMIT_NOFILE)
new_limit = 128_000
resource.setrlimit(resource.RLIMIT_NOFILE, (new_limit, hard_limit))
print(f"initial max no_fds: {soft_limit}, now set to {new_limit}")


class ModelEndpoint(NamedTuple):
    url: str
    model_name: str


def get_worker_list(
    url: str, model_name: str, raw_url: bool = False
) -> List[ModelEndpoint]:
    response = requests.post(url + "/workers/list", json={"model_name": model_name})

    if response.status_code == 200:
        ep = (
            ""
            if raw_url
            else (EXEC_ENDPOINT if model_name == "code_execution" else CHAT_ENDPOINT)
        )
        return [
            ModelEndpoint(url=worker + ep, model_name=model_name)
            for worker in response.json()["workers"]
        ]
    else:
        raise ValueError(
            f"list workers get code {response.status_code}: {response.json()}"
        )


def test_worker(
    url: str, model_name: str, interval: int = 60, max_tries: int = 6
) -> bool:
    if model_name == "code_execution":
        return test_exec_worker(url, interval, max_tries)
    else:
        return test_model_worker(url, model_name, interval, max_tries)


def test_exec_worker(url: str, interval: int, max_tries: int) -> bool:
    headers = {"Content-Type": "application/json"}
    data = {"code": "assert 1+1 == 2"}

    print(f"Testing connections to {url} for code_execution", flush=True)
    for attempt in range(max_tries):
        try:
            print(f" {attempt + 1}", end="", flush=True)
            response = requests.post(
                f"{url}{EXEC_ENDPOINT}", headers=headers, json=data
            )
            response.raise_for_status()  # Raises an HTTPError for bad responses
            print(f"Test successful for {url}!")
            return True
        except Exception as e:
            if attempt < max_tries - 1:
                time.sleep(interval)
                interval *= 2
            else:
                print(f"Maximum attempts reached. Test failed for {url}!")

    return False


def test_model_worker(
    url: str, model_name: str, interval: int = 60, max_tries: int = 6
) -> bool:
    """Since it might take sometime for the worker to be running."""

    headers = {"Content-Type": "application/json"}
    data = {
        "model": model_name,
        "prompt": "San Francisco is a",
        "max_tokens": 7,
        "temperature": 0,
    }

    print(f"Testing connections to {url} for {model_name}", flush=True)
    for attempt in range(max_tries):
        try:
            print(f" {attempt + 1}", end="", flush=True)
            response = requests.post(
                f"{url}{COMP_ENDPOINT}", headers=headers, json=data, timeout=60
            )
            response.raise_for_status()  # Raises an HTTPError for bad responses
            print(f"Test successful for {url}!")
            return True
        except Exception as e:
            print(f"get exeception {str(e)}", flush=True)
            if attempt < max_tries - 1:
                time.sleep(interval)
                interval *= 2
            else:
                print(f"Maximum attempts reached. Test failed for {url}!")

    return False
    
# test_model_worker("http://10.200.85.63:5000", "meta-llama/Meta-Llama-3.1-405B-Instruct-FP8")

async def _http_post(
    session: aiohttp.ClientSession,
    endpoint: ModelEndpoint,
    data: Dict[str, Any],
) -> Dict[str, Any]:
    headers = {"Content-Type": "application/json"}

    response = await session.post(
        endpoint.url,
        headers=headers,
        json=data,
    )

    if response.status == 200:
        try:
            response_json = await response.json()
            return {"status_ok": True, "info": response_json}
        except aiohttp.ClientResponseError as e:
            return {"status_ok": False, "err_str": f"Failed to parse {str(e)}"}
    else:
        return {"status_ok": False, "err_str": f"Response status {response.status}"}


async def http_post_with_retry(
    session: aiohttp.ClientSession,
    endpoint: ModelEndpoint,
    data: Dict[str, Any],
    timeout: int = DEFAULT_REQUEST_TIMEOUT,
    max_retry: int = 3,
) -> Dict[str, Any]:

    retry = 0
    err_str = "ERROR: "
    while retry < max_retry:
        try:
            return_dict = await asyncio.wait_for(
                _http_post(session, endpoint, data), timeout
            )

            if return_dict["status_ok"]:
                return return_dict
            else:
                err_str += f"[{retry}] {return_dict['err_str']}; "
                retry += 1

        except asyncio.TimeoutError:
            err_str += f"[{retry}] timeout; "
            retry += 1
        except Exception as e:
            # FIXME: it could also be timeout on the server side
            err_str += f"[{retry}] {str(e)}; "
            retry += 1

    return {"status_ok": False, "info": {"error": err_str}}
