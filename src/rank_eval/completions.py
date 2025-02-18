import logging
import os
import requests
import time
from collections import defaultdict
import utils

__all__ = ["vllm_endpoint_chat_completions"]

CHAT_ENDPOINT: str = "/v1/chat/completions"

global client
client = None

def vllm_endpoint_chat_completions(
    messages_batch: list[list[dict]],
    model_name="test",
    max_new_tokens: int = None,
    temperature: float = 1.0,
    request_timeout: int = 10,
    max_retry: int = 3,
    model_endpoint: str = "http://10.200.74.98:5001",
    price_per_token: float = 9e-7,
    model_kwargs=None,
    **decoding_kwargs,
) -> dict[str, list]:
    n_examples = len(messages_batch)
    responses = []
    
    if isinstance(messages_batch[0], dict):
        messages_batch = [messages_batch]
    usage_total = defaultdict(int)
    
    with utils.Timer() as t:
        for messages in messages_batch:
            status_ok = False
            request_data = {
                "model": model_name,
                "messages": messages,
                "temperature": temperature,
            }
            if max_new_tokens is not None: request_data["max_tokens"] = max_new_tokens
            headers = {"Content-Type": "application/json"}
            # retry = 0
        
            for _ in range(max_retry):
                # retry += 1
                # send the request with aiohttp
                try:
                    http_response = requests.post(
                        f"{model_endpoint}{CHAT_ENDPOINT}",
                        headers=headers,
                        json=request_data,
                        timeout=request_timeout,
                    )
                except:
                    logging.error(f"Failed to send request, retrying after {request_timeout} seconds...")
                    time.sleep(request_timeout)
                    continue

                if http_response.status_code == 200:
                    response = http_response.json()
                    response_text = response["choices"][0]["message"]['content']
                    usage = response["usage"]
                    for k, v in usage.items(): usage_total[k] += v if v is not None else 0
                    if "m" == response_text[-1].lower(): # specific filter for the ranking eval prompt
                        status_ok = True
                        responses.append(response_text)
                        break
                logging.error(f"Failed to generate chat completion: {http_response.text}, retrying after {request_timeout} seconds...")
                time.sleep(request_timeout)
                
            try:
                if not status_ok:
                    logging.error(f"Failed to generate chat completion: {http_response.text}, max retries ({max_retry}) reached.")
                    responses.append(None)
            except:
                logging.error(f"Failed to generate chat completion due to http request errors, max retries ({max_retry}) reached.")
                responses.append(None)

    avg_time = [t.duration / n_examples] * len(responses)
    price_per_example = [price_per_token] * len(responses)
    usage_per = {k: v / n_examples for k, v in usage_total.items()}
    return dict(
        completions=responses, 
        price_per_example=price_per_example, 
        time_per_example=avg_time,
        usage_per_example=usage_per,
        usage_total=usage_total,
    )

def openai_chat_completions(
    messages_batch: list[list[dict]],
    model_name="test",
    max_new_tokens: int = None,
    temperature: float = 1.0,
    request_timeout: int = 10,
    max_retry: int = 1,
    api_account: str = "openai",
    price_per_token: float = 9e-7,
    model_kwargs=None,
    **decoding_kwargs,
) -> dict[str, list]:

    from openai import OpenAI
    from keys import API_KEY

    global client
    if not client:
        client = OpenAI(
            api_key = API_KEY.get(api_account, os.getenv("OPENAI_API_KEY")), 
        )
    n_examples = len(messages_batch)
    responses = []
    usage_total = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    with utils.Timer() as t:
        for messages in messages_batch:
            if "o3" in model_name:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=temperature,
                    max_completion_tokens=max_new_tokens,
                )
                response_text = response.choices[0].message.content.strip()
                if response_text == "":
                    response = client.chat.completions.create(
                        reasoning_effort="low",
                        model=model_name,
                        messages=messages,
                        temperature=temperature,
                        max_completion_tokens=max_new_tokens,
                    )
                    response_text = response.choices[0].message.content.strip()
                    if response_text == "":
                        response = client.chat.completions.create(
                            reasoning_effort="low",
                            model=model_name,
                            messages=messages,
                            temperature=temperature,
                            max_completion_tokens=max_new_tokens*2,
                        )
                        response_text = response.choices[0].message.content.strip()
            else:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_new_tokens,
                )
                response_text = response.choices[0].message.content.strip()
            usage = response.usage
            usage_total["prompt_tokens"] += usage.prompt_tokens
            usage_total["completion_tokens"] += usage.completion_tokens
            if "o3" in model_name:
                usage_total["total_tokens"] += usage.total_tokens
            usage_total["total_tokens"] += usage.total_tokens
            responses.append(response_text)
    avg_time = [t.duration / n_examples] * len(responses)

    if "o3" in model_name:
        price = usage_total["prompt_tokens"] * 1.1 / 1000000 + usage_total["completion_tokens"] * 4.4 / 1000000
    if "4o" in model_name:
        price = usage_total["prompt_tokens"] * 2.5 / 1000000 + usage_total["completion_tokens"] * 10 / 1000000
    
    price_per_example = price
    usage_per = {k: v / n_examples for k, v in usage_total.items()}
    return dict(
        completions=responses,
        price_per_example=price_per_example,
        time_per_example=avg_time,
        usage_per_example=usage_per,
        usage_total=usage_total,
    )