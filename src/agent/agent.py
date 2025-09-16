import typing as t
import asyncio
from abc import ABC, abstractmethod
import os

# External Libraries
import requests
import json
from together import Together
from openai import AsyncOpenAI
import aiohttp # Using aiohttp for async HTTP requests

# A standard response type to avoid type errors with Together.
try:
    from together.types.chat_completions import ChatCompletionResponse
except ImportError:
    ChatCompletionResponse = t.Any


# --- ABSTRACT BASE CLASS (The Core Abstraction) ---
class BaseLLMClient(ABC):
    """
    Abstract base class for all LLM clients.
    Defines a common, asynchronous interface for calling LLMs.
    """
    def __init__(self, model: str, **kwargs):
        self.client = None
        self.model = model
        self.kwargs = kwargs
        
    @abstractmethod
    async def __call__(
        self,
        prompt: str,
        max_tokens: int = 1_000,
        system_prompt: t.Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Defines the async call method for all concrete clients.
        """
        ...
        
    def _create_messages(
        self,
        prompt: str,
        system_prompt: t.Optional[str] = None
    ) -> t.List[t.Dict[str, str]]:
        """
        Helper to create a standard message dictionary with an optional system prompt.
        The system prompt is always added as the first message to set the model's context.
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return messages


# --- CONCRETE IMPLEMENTATIONS (The Adapters) ---

class OpenAIClient(BaseLLMClient):
    """
    Adapter for the OpenAI (and OpenAI-compatible) Async API client.
    """
    def __init__(self, model: str, **kwargs):
        super().__init__(model, **kwargs)
        self.client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    async def __call__(
        self,
        prompt: str,
        max_tokens: int = 1_000,
        system_prompt: t.Optional[str] = None,
        **kwargs
    ) -> str:
        try:
            messages = self._create_messages(prompt, system_prompt)
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                **self.kwargs,
                **kwargs,
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error from OpenAI: {e}"


class TogetherAIClient(BaseLLMClient):
    """
    Adapter for the Together API client.
    Uses asyncio.to_thread to run the synchronous client in a separate thread.
    """
    def __init__(self, model: str, **kwargs):
        super().__init__(model, **kwargs)
        # Note: Together() automatically looks for TOGETHER_API_KEY env var
        self.client = Together() 

    async def __call__(
        self,
        prompt: str,
        max_tokens: int = 1_000,
        system_prompt: t.Optional[str] = None,
        **kwargs
    ) -> str:
        # Use asyncio.to_thread to run the synchronous Together client
        # without blocking the event loop.
        try:
            messages = self._create_messages(prompt, system_prompt)
            response: ChatCompletionResponse = await asyncio.to_thread(
                self.client.chat.completions.create,
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                **self.kwargs,
                **kwargs,
            )
            return str(response.choices[0].message.content)
        except Exception as e:
            return f"Error from TogetherAI: {e}"


class GeminiClient(BaseLLMClient):
    """
    Adapter for the Gemini REST API, using aiohttp for async HTTP requests.
    """
    def __init__(self, model: str, **kwargs):
        super().__init__(model, **kwargs)
        self.api_key = os.getenv("GEMINI_API_KEY")
        self.url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={self.api_key}"

    async def __call__(
        self,
        prompt: str,
        max_tokens: int = 1_000,
        system_prompt: t.Optional[str] = None,
        **kwargs
    ) -> str:
        if not self.api_key:
            return "Error: GEMINI_API_KEY not found."

        contents = self._create_messages(prompt, system_prompt)
        payload = {
            "contents": contents,
            "generationConfig": {"maxOutputTokens": max_tokens},
            **self.kwargs,
            **kwargs,
        }
        
        headers = {"Content-Type": "application/json"}
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(self.url, headers=headers, json=payload) as response:
                    response.raise_for_status()
                    response_data = await response.json()
                    return response_data['candidates'][0]['content']['parts'][0]['text']
        except aiohttp.ClientError as e:
            return f"Error from Gemini (requests): {e}"
        except (KeyError, IndexError) as e:
            return f"Error parsing Gemini response: {e}"


class GroqClient(BaseLLMClient):
    """
    Adapter for the Groq REST API, using aiohttp for async HTTP requests.
    """
    def __init__(self, model: str, **kwargs):
        super().__init__(model, **kwargs)
        self.api_key = os.getenv("GROQ_API_KEY")
        self.url = "https://api.groq.com/openai/v1/chat/completions"

    async def __call__(
        self,
        prompt: str,
        max_tokens: int = 1_000,
        system_prompt: t.Optional[str] = None,
        **kwargs
    ) -> str:
        if not self.api_key:
            return "Error: GROQ_API_KEY not found."

        messages = self._create_messages(prompt, system_prompt)
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            **self.kwargs,
            **kwargs
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(self.url, headers=headers, json=payload) as response:
                    response.raise_for_status()
                    response_data = await response.json()
                    return response_data['choices'][0]['message']['content']
        except aiohttp.ClientError as e:
            return f"Error from Groq (requests): {e}"
        except (KeyError, IndexError) as e:
            return f"Error parsing Groq response: {e}"


class MistralClient(BaseLLMClient):
    """
    Adapter for the Mistral REST API, using aiohttp for async HTTP requests.
    """
    def __init__(self, model: str, **kwargs):
        super().__init__(model, **kwargs)
        self.api_key = os.getenv("MISTRAL_API_KEY")
        self.url = "https://api.mistral.ai/v1/chat/completions"

    async def __call__(
        self,
        prompt: str,
        max_tokens: int = 1_000,
        system_prompt: t.Optional[str] = None,
        **kwargs
    ) -> str:
        if not self.api_key:
            return "Error: MISTRAL_API_KEY not found."

        messages = self._create_messages(prompt, system_prompt)
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            **self.kwargs,
            **kwargs
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(self.url, headers=headers, json=payload) as response:
                    response.raise_for_status()
                    response_data = await response.json()
                    return response_data['choices'][0]['message']['content']
        except aiohttp.ClientError as e:
            return f"Error from Mistral (requests): {e}"
        except (KeyError, IndexError) as e:
            return f"Error parsing Mistral response: {e}"

        
# ('openai', OpenAIClient(model="gpt-3.5-turbo")),
# ('togetherai', TogetherAIClient(model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free")),
# ('gemini', GeminiClient(model="gemini-1.5-flash-latest")),
# ('groq', GroqClient(model="llama3-8b-8192")),
# ('mistral', MistralClient(model="mistral-tiny")),