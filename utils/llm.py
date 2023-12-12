from typing import Any, List, Mapping, Optional
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM
import requests
import json
from textwrap import dedent

class CustomLLM(LLM):
    api_url: Optional[str] = "http://localhost:8080/completion"
    max_new_tokens: Optional[int] = 1024
    top_p: Optional[float] = 0.9
    temperature: Optional[float] = 0
    repetition_penalty: Optional[float] = 1.5
    custom_kwargs: Optional[Mapping[str, Any]] = {}

    def __init__(
        self,
        **kwargs: Any,
    ):
        super().__init__(**kwargs) 
        self.api_url = kwargs.get("api_url", self.api_url)
        self.max_new_tokens = kwargs.get("max_new_tokens", self.max_new_tokens)
        self.top_p = kwargs.get("top_p", self.top_p)
        self.temperature = kwargs.get("temperature", self.temperature)
        self.repetition_penalty = kwargs.get("repetition_penalty", self.repetition_penalty)
        self.custom_kwargs = kwargs 

    @property
    def _llm_type(self) -> str:
        return "custom"

    async def _acall(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
    ) -> str:
        if stop is not None:
            raise ValueError("stop kwargs are not permitted.")

        json_body = {
            "prompt": prompt,
            "max_new_tokens": self.max_new_tokens,
            "top_p": self.top_p,
            "temperature": self.temperature,
            "repetition_penalty": self.repetition_penalty,
            **self.custom_kwargs,
        }
        data = json.dumps(json_body)
        response = requests.request("POST", self.api_url, data=data)
        response = json.loads(response.content.decode("utf-8"))
        return response['content']

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
    ) -> str:
        if stop is not None:
            raise ValueError("stop kwargs are not permitted.")

        json_body = {
            "prompt": prompt,
            "max_new_tokens": self.max_new_tokens,
            "top_p": self.top_p,
            "temperature": self.temperature,
            "repetition_penalty": self.repetition_penalty,
            **self.custom_kwargs,
        }
        data = json.dumps(json_body)
        response = requests.request("POST", self.api_url, data=data)
        response = json.loads(response.content.decode("utf-8"))
        return response['content']

    @property
    def _identifying_params(self, **kwargs: Any) -> Mapping[str, Any]:
        identifying_params = {
            "api_url": self.api_url,
            "max_new_tokens": self.max_new_tokens,
            "top_p": self.top_p,
            "temperature": self.temperature,
            "repetition_penalty": self.repetition_penalty,
            **self.custom_kwargs,
        }
        return identifying_params
    
    def format_prompt(self, user_query: str, system_instruction: str = "be honest and truthful.", llm_answer_start:str = "") -> str:
        prompt = f"""[INST]<<SYS>>
            {system_instruction}
            <<SYS>>
            {user_query}[/INST]
            {llm_answer_start}
            """.replace('\n', ' ').strip()
        
        return prompt
    

if __name__ == "__main__":
    llm = CustomLLM(temperature=0.7)
    prompt = llm.format_prompt("Who are you?",system_instruction="You are a pirate. You only answer questions in pirate english.")
    response = llm(prompt)
    print(response)