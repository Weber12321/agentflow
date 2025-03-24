from abc import ABC, abstractmethod
from typing import List, Dict, Optional, AsyncIterator


class BaseLLMClient(ABC):
    """
    Abstract base class for LLM clients.  Defines the common interface.
    """

    @abstractmethod
    def inference(self, prompt: str, **kwargs) -> str:
        """
        Performs synchronous inference.

        Args:
            prompt: The input prompt.
            **kwargs:  Additional model-specific parameters (e.g., temperature, max_tokens).

        Returns:
            The generated text.
        """
        pass

    @abstractmethod
    async def async_inference(self, prompt: str, **kwargs) -> str:
        """
        Performs asynchronous inference.

        Args:
            prompt: The input prompt.
            **kwargs: Additional model-specific parameters.

        Returns:
            The generated text.
        """
        pass

    @abstractmethod
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """
        Performs synchronous chat completion.

        Args:
            messages: A list of messages in the conversation.  Each message
                      is a dictionary with "role" (e.g., "user", "assistant", "system")
                      and "content" keys.
            **kwargs:  Additional model-specific parameters.

        Returns:
            The generated response.
        """
        pass

    @abstractmethod
    async def async_chat(
        self, messages: List[Dict[str, str]], **kwargs
    ) -> str:
        """
        Performs asynchronous chat completion.

        Args:
            messages: A list of messages in the conversation.
            **kwargs: Additional model-specific parameters.

        Returns:
            The generated response.
        """
        pass

    # Optional:  Streaming support (synchronous and asynchronous)
    def stream_inference(self, prompt: str, **kwargs) -> Optional[str]:
        """
        Performs synchronous streaming inference (if supported by the API).

        Args:
            prompt: The input prompt.
            **kwargs: Additional model-specific parameters.

        Yields:
           Chunks of the generated text as they become available.
           Returns None by default if not implemented
        """
        return None

    async def async_stream_inference(
        self, prompt: str, **kwargs
    ) -> Optional[AsyncIterator[str]]:
        """
        Performs asynchronous streaming inference (if supported).

        Args:
            prompt: The input prompt.
            **kwargs: Additional model-specific parameters.

        Yields:
            Chunks of the generated text as they become available.
            Returns None by default if not implemented
        """
        return None

    def stream_chat(
        self, messages: List[Dict[str, str]], **kwargs
    ) -> Optional[str]:
        """
        Streams chat responses (if supported).

        Args:
            messages: List of chat messages.
            **kwargs:  Additional parameters.

        Yields:
            Chunks of the generated text.
            Returns None by default if not implemented
        """
        return None

    async def async_stream_chat(
        self, messages: List[Dict[str, str]], **kwargs
    ) -> Optional[AsyncIterator[str]]:
        """
        Asynchronously streams chat responses (if supported).

        Args:
            messages: List of chat messages.
            **kwargs: Additional parameters.

        Yields:
            Chunks of the generated text.
            Returns None by default if not implemented
        """
        return None

    # Optional:  Methods for managing API keys, endpoints, etc.
    def set_api_key(self, api_key: str):
        """Sets the API key (if applicable)."""
        pass  # Implement in concrete classes

    def get_model_name(self) -> str:
        return ""
