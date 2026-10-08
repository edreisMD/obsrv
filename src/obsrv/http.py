"""Small injectable HTTP transport. Never persist headers, response bodies or URL errors."""

import json
import os
import urllib.request
from typing import Protocol


class Transport(Protocol):
    def get(self, url: str, timeout: float) -> str: ...
    def post(self, url: str, body: dict, timeout: float) -> None: ...


class ProfileRejectedError(RuntimeError):
    """Engine definitively rejected this operation; response body is not retained."""


class HttpTransport:
    def __init__(self, auth_env: str | None = None):
        self.auth_env = auth_env
        # Do not forward credentials through redirects.
        self.opener = urllib.request.build_opener(_NoRedirect())

    def _request(self, url, timeout, body=None):
        headers = {"Content-Type": "application/json"} if body is not None else {}
        if self.auth_env:
            token = os.environ[self.auth_env]
            headers["Authorization"] = f"Bearer {token}"
        req = urllib.request.Request(
            url,
            data=json.dumps(body).encode() if body is not None else None,
            headers=headers,
            method="POST" if body is not None else "GET",
        )
        with self.opener.open(req, timeout=timeout) as response:
            payload = response.read(16 * 1024 * 1024 + 1)
            if len(payload) > 16 * 1024 * 1024:
                raise ValueError("HTTP response exceeds collector size limit")
            return payload.decode()

    def get(self, url, timeout):
        return self._request(url, timeout)

    def post(self, url, body, timeout):
        response = self._request(url, timeout, body)
        if response.strip():
            try:
                result = json.loads(response)
            except ValueError:
                return
            if isinstance(result, dict) and result.get("success") is False:
                raise ProfileRejectedError("engine rejected profile operation")


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None
