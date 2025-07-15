# Name: HttpsApi
# Parameters:
# host:
# key:
# model:
# timeout: 20

from __future__ import annotations
import http.client
import json
import time
from typing import Any
import traceback
from ...base import LLM
from threading import Lock

class HttpsApi(LLM):
    _lock = Lock()
    _last_request_time = 0
    _request_count = 0

    def __init__(self, host, key, model, timeout=20, **kwargs):
        """Https API
        Args:
            host   : host name. please note that the host name does not include 'https://'
            key    : API key.
            model  : LLM model name.
            timeout: API timeout.
        """
        super().__init__(**kwargs)
        self._host = host
        self._key = key
        self._model = model
        self._timeout = timeout
        self._kwargs = kwargs
        self._cumulative_error = 0

    def _rate_limit(self):
        with self._lock:
            current_time = time.time()
            if self._request_count >= 10:
                elapsed_time = current_time - self._last_request_time
                if elapsed_time < 60:
                    sleep_time = 60 - elapsed_time
                    print(f"Rate limit reached. Sleeping for {sleep_time:.2f} seconds.")
                    time.sleep(sleep_time)
                self._request_count = 0
                self._last_request_time = time.time()
            else:
                self._request_count += 1

    def draw_sample(self, prompt: str | Any, *args, **kwargs) -> str:
        if isinstance(prompt, str):
            prompt = [{'role': 'user', 'content': prompt.strip()}]

        while True:
            try:
                self._rate_limit()
                conn = http.client.HTTPSConnection(self._host, timeout=self._timeout)
                payload = json.dumps({
                    'max_tokens': self._kwargs.get('max_tokens', 8192),
                    'top_p': self._kwargs.get('top_p', None),
                    'temperature': self._kwargs.get('temperature', 0.7),
                    'model': self._model,
                    'messages': prompt
                })
                headers = {
                    'Authorization': f'Bearer {self._key}',
                    'User-Agent': 'Apifox/1.0.0 (https://apifox.com)',
                    'Content-Type': 'application/json'
                }
                conn.request('POST', '/v1/chat/completions', payload, headers)
                res = conn.getresponse()
                data = res.read().decode('utf-8')
                data = json.loads(data)
                response = data['choices'][0]['message']['content']
                if self.debug_mode:
                    self._cumulative_error = 0
                return response
            except Exception as e:
                self._cumulative_error += 1
                if self.debug_mode:
                    if self._cumulative_error == 10:
                        raise RuntimeError(f'{self.__class__.__name__} error: {traceback.format_exc()}.'
                                           f'You may check your API host and API key.')
                else:
                    print(f'{self.__class__.__name__} error: {traceback.format_exc()}.'
                          f'You may check your API host and API key.')
                    time.sleep(2)
                continue
