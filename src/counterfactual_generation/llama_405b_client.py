import httpx
import requests
import time


CHAT_ENDPOINT: str = "/v1/chat/completions"
EXEC_ENDPOINT: str = "/exec"
COMP_ENDPOINT: str = "/v1/completions"

class Llama3_Client(httpx.Client):
    def __init__(self, model_endpoint, model_name, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_endpoint = model_endpoint


    def chat_completion(self, messages, 
                        model_name, 
                        max_tokens: int = 512, 
                        temperature: float = 1.0, 
                        request_timeout: int = 10,
                        max_retry: int = 3):
        request_data = {
            "model": model_name,
            "messages": messages,
            "temperature": temperature,
        }
        if max_tokens is not None:
            request_data["max_tokens"] = max_tokens
        
        headers = {"Content-Type": "application/json"}

        retry = 0
        while retry < max_retry:
            # send the request with aiohttp
            http_response = requests.post(
                f"{self.model_endpoint}{CHAT_ENDPOINT}",
                headers=headers,
                json=request_data,
            )

            if http_response.status_code == 200:
                response = http_response.json()
                response_text = response["choices"][0]["message"]['content']
                usage = response["usage"]
                return {"status_ok": True, "content": response_text, "usage": usage}
            print(f"Failed to generate next turn: {http_response.text}, retrying...")
            retry += 1
            time.sleep(request_timeout)
        print(f"Failed to generate next turn: {http_response.text}, max retries ({max_retry}) reached.")
        return {"status_ok": False, "content": None, "usage": None}


    def completion(self, messages, 
                        model_name, 
                        max_tokens: int = 512, 
                        temperature: float = 1.0, 
                        request_timeout: int = 10,
                        max_retry: int = 3):
        
        if self.tokenizer == None: 
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        message_prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        request_data = {
            "model": model_name,
            "prompt": message_prompt,
            "temperature": temperature,
        }
        if max_tokens is not None:
            request_data["max_tokens"] = max_tokens
        
        headers = {"Content-Type": "application/json"}

        retry = 0
        while retry < max_retry:
            # send the request with aiohttp
            http_response = requests.post(
                f"{self.model_endpoint}{COMP_ENDPOINT}",
                headers=headers,
                json=request_data,
            )

            if http_response.status_code == 200:
                response = http_response.json()
                response_text = response["choices"][0]["text"]
                usage = response["usage"]
                return {"status_ok": True, "content": response_text, "usage": usage}
            print(f"Failed to generate next turn: {http_response.text}, retrying...")
            retry += 1
            time.sleep(request_timeout)
        print(f"Failed to generate next turn: {http_response.text}, max retries ({max_retry}) reached.")
        return {"status_ok": False, "content": None, "usage": None}