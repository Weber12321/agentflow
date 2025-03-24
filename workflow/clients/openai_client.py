import logging
from typing import AsyncIterator, Dict, List
from openai import OpenAI, AsyncOpenAI
from .base import BaseLLMClient


class OpenAIClient(BaseLLMClient):
    """
    Client for interacting with the OpenAI API.
    """

    def __init__(
        self, 
        api_key: str, 
        model_name: str = "gpt-3.5-turbo", 
        **kwargs
    ):
        if kwargs.get("local", False) and kwargs.get("base_url", ""):
            logging.info(
                "Using the provided OpenAI API endpoint %s", kwargs.get(
                    "base_url"
                )
            )
            self.client = OpenAI(
                api_key=api_key, base_url=kwargs.get("base_url")
            )
            self.async_client = AsyncOpenAI(
                api_key=api_key, base_url=kwargs.get("base_url")
            )
        else:
            logging.info("Using the default OpenAI API endpoint.")
            self.client = OpenAI(api_key=api_key)
            self.async_client = AsyncOpenAI(api_key=api_key)
        
        self.model_name = model_name
        self.default_kwargs = kwargs

    def inference(self, prompt: str, **kwargs) -> str:
        combined_kwargs = {
            **self.default_kwargs,
            **kwargs,
        }  # combine default and call kwargs.
        response = self.client.completions.create(
            model=self.model_name, prompt=prompt, **combined_kwargs
        )
        return response.choices[0].text

    async def async_inference(self, prompt: str, **kwargs) -> str:
        combined_kwargs = {**self.default_kwargs, **kwargs}
        response = await self.async_client.completions.create(
            model=self.model_name, prompt=prompt, **combined_kwargs
        )
        return response.choices[0].text

    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        combined_kwargs = {**self.default_kwargs, **kwargs}
        response = self.client.chat.completions.create(
            model=self.model_name, messages=messages, **combined_kwargs
        )
        return response.choices[0].message.content

    async def async_chat(
        self, messages: List[Dict[str, str]], **kwargs
    ) -> str:
        combined_kwargs = {**self.default_kwargs, **kwargs}
        response = await self.async_client.chat.completions.create(
            model=self.model_name, messages=messages, **combined_kwargs
        )
        return response.choices[0].message.content

    def stream_inference(self, prompt: str, **kwargs) -> str:
        combined_kwargs = {**self.default_kwargs, **kwargs}
        response_stream = self.client.completions.create(
            model=self.model_name,
            prompt=prompt,
            stream=True,
            **combined_kwargs
        )
        for response in response_stream:
            yield response.choices[0].text

    async def async_stream_inference(
        self, prompt: str, **kwargs
    ) -> AsyncIterator[str]:
        combined_kwargs = {**self.default_kwargs, **kwargs}
        response_stream = await self.async_client.completions.create(
            model=self.model_name,
            prompt=prompt,
            stream=True,
            **combined_kwargs
        )
        async for response in response_stream:
            yield response.choices[0].text

    def stream_chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        combined_kwargs = {**self.default_kwargs, **kwargs}
        response_stream = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            stream=True,
            **combined_kwargs
        )
        for response in response_stream:
            yield response.choices[0].delta.content or ""

    async def async_stream_chat(
        self, messages: List[Dict[str, str]], **kwargs
    ) -> AsyncIterator[str]:
        combined_kwargs = {**self.default_kwargs, **kwargs}
        response_stream = await self.async_client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            stream=True,
            **combined_kwargs
        )
        async for response in response_stream:
            yield response.choices[0].delta.content or ""

    def set_api_key(self, api_key: str):
        self.client = OpenAI(api_key=api_key)
        self.async_client = AsyncOpenAI(api_key=api_key)

    def get_model_name(self) -> str:
        return self.model_name
