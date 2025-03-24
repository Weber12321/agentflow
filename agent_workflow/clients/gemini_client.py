import logging
import google.generativeai as genai
from typing import AsyncIterator, Dict, List

from .base import BaseLLMClient


class GeminiClient(BaseLLMClient):
    """
    Client for interacting with Google's Gemini API.
    """

    def __init__(self, api_key: str, model_name: str = "gemini-pro", **kwargs):

        genai.configure(api_key=api_key)
        self.model_name = model_name
        self.model = genai.GenerativeModel(model_name)
        self.default_kwargs = kwargs

    def inference(self, prompt: str, **kwargs) -> str:
        combined_kwargs = {**self.default_kwargs, **kwargs}
        response = self.model.generate_content(prompt, **combined_kwargs)
        return response.text

    async def async_inference(self, prompt: str, **kwargs) -> str:
        combined_kwargs = {**self.default_kwargs, **kwargs}
        # Use asyncio.to_thread for synchronous methods in async context
        response = await asyncio.to_thread(
            self.model.generate_content, prompt, **combined_kwargs
        )
        return response.text

    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        combined_kwargs = {**self.default_kwargs, **kwargs}
        # Convert Langchain style messages to Gemini style
        chat = self.model.start_chat(history=self._convert_message(messages))
        response = chat.send_message(
            messages[-1]["content"], **combined_kwargs
        )  # use the last message.
        return response.text

    async def async_chat(
        self, messages: List[Dict[str, str]], **kwargs
    ) -> str:
        combined_kwargs = {**self.default_kwargs, **kwargs}
        chat = self.model.start_chat(history=self._convert_message(messages))
        # Use asyncio.to_thread for synchronous method.
        response = await asyncio.to_thread(
            chat.send_message, messages[-1]["content"], **combined_kwargs
        )
        return response.text

    def stream_inference(self, prompt: str, **kwargs) -> str:
        combined_kwargs = {**self.default_kwargs, **kwargs}
        response_stream = self.model.generate_content(
            prompt, stream=True, **combined_kwargs
        )
        for chunk in response_stream:
            yield chunk.text

    async def async_stream_inference(
        self, prompt: str, **kwargs
    ) -> AsyncIterator[str]:
        combined_kwargs = {**self.default_kwargs, **kwargs}
        response_stream = await asyncio.to_thread(
            self.model.generate_content, prompt, stream=True, **combined_kwargs
        )
        # response_stream = self.model.generate_content(prompt, stream=True)
        async for chunk in response_stream:
            yield chunk.text

    def stream_chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        combined_kwargs = {**self.default_kwargs, **kwargs}
        chat = self.model.start_chat(history=self._convert_message(messages))
        response_stream = chat.send_message(
            messages[-1]["content"], stream=True, **combined_kwargs
        )
        for chunk in response_stream:
            yield chunk.text

    async def async_stream_chat(
        self, messages: List[Dict[str, str]], **kwargs
    ) -> AsyncIterator[str]:
        combined_kwargs = {**self.default_kwargs, **kwargs}
        chat = self.model.start_chat(history=self._convert_message(messages))
        response_stream = await asyncio.to_thread(
            chat.send_message,
            messages[-1]["content"],
            stream=True,
            **combined_kwargs
        )
        async for chunk in response_stream:
            yield chunk.text

    def set_api_key(self, api_key: str):
        import google.generativeai as genai

        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(self.model_name)  # reset the model

    def _convert_message(
        self, messages: List[Dict[str, str]]
    ) -> List[Dict[str, str]]:
        history = []
        role_mapping = {
            "user": "user",
            "assistant": "model",
            "system": "user",  # there is no system role in gemini api.
        }
        for message in messages:
            role = role_mapping.get(message["role"], "user")
            if role == "model":
                # Gemini chat api need strictly alternating user/model message.
                history.append({"role": "user", "parts": [message["content"]]})
                history.append(
                    {"role": "model", "parts": [message["content"]]}
                )
            else:
                history.append({"role": role, "parts": [message["content"]]})
        return history[
            :-1
        ]  # remove the last one sicne we only need history to start the chat.

    def get_model_name(self) -> str:
        return self.model_name
