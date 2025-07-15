from __future__ import annotations
import openai
from typing import Any
from ...base import LLM
from pydantic import BaseModel
from openai import OpenAI

class Cluster(BaseModel):
    Group: list[list[int]]


class HttpsApiOpenAI4Cluster(LLM):
    def __init__(self, base_url: str, api_key: str, model: str, timeout=30, **kwargs):
        super().__init__()
        self._model = model
        self._client = openai.OpenAI(api_key=api_key, timeout=timeout, **kwargs)

    def draw_sample(self, prompt: str | Any, *args, **kwargs) -> str:
        try:
            if isinstance(prompt, str):
                messages = [{'role': 'system', 'content': "You are an expert in program analysis and logic abstraction."},
                            {"role": "user", "content": prompt}]
            elif isinstance(prompt, list):
                messages = prompt
            else:
                raise ValueError("Unsupported prompt format")
            print("Hi")
            response = self._client.chat.completions.parse(
                model=self._model,
                messages=messages,
                response_format=Cluster,
            )
            return response.choices[0].message.parsed.Group
        except Exception as e:
            print(f"Error in OpenAI API call: {e}")
            return None


