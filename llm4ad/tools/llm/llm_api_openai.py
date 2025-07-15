from __future__ import annotations
import openai
from typing import Any
from ...base import LLM


class HttpsApiOpenAI(LLM):
    def __init__(self, base_url: str, api_key: str, model: str, timeout=30, **kwargs):
        super().__init__()
        self._model = model
        self._client = openai.OpenAI(api_key=api_key, timeout=timeout, **kwargs)

    def draw_sample(self, prompt: str | Any, *args, **kwargs) -> str:
        try:
            if isinstance(prompt, str):
                messages = [{'role': 'system', 'content': "You are an expert in the domain of optimization heuristics helping to design heuristics that can effectively solve optimization problems."},
                            {"role": "user", "content": prompt}]
            elif isinstance(prompt, list):
                messages = prompt
            else:
                raise ValueError("Unsupported prompt format")

            response = self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                stream=False,
                max_tokens=8192,
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error in OpenAI API call: {e}")
            return None



