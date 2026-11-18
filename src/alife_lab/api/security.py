"""Local-only access policy: Host allow-list, Origin checks, session token, body limits.

Threat model (spec 17.1): the server binds to loopback, but any web page open in
the user's browser can send requests to localhost. Defences:

* Host header allow-list -> blocks DNS-rebinding.
* State-changing requests require ``X-ALife-Token`` (a per-start random token
  injected into the served index.html). Cross-origin pages cannot read it and
  cannot attach custom headers without a CORS preflight, which we never grant.
* If an ``Origin`` header is present it must be one of the server's own origins.
* WebSocket upgrades require an allowed Origin and the token as a query param.
* Request bodies are capped.
"""

from __future__ import annotations

import hmac
import json
import os
import secrets
from collections.abc import Awaitable, Callable, MutableMapping
from pathlib import Path
from typing import Any

Scope = MutableMapping[str, Any]
Message = MutableMapping[str, Any]
Receive = Callable[[], Awaitable[Message]]
Send = Callable[[Message], Awaitable[None]]
ASGIApp = Callable[[Scope, Receive, Send], Awaitable[None]]

TOKEN_HEADER = "x-alife-token"  # noqa: S105 - header name, not a secret
MAX_BODY_BYTES = 2 * 1024 * 1024
SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})
LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})


def is_loopback(host: str) -> bool:
    return host in LOOPBACK_HOSTS


def make_session_token(path: Path) -> str:
    token = secrets.token_urlsafe(32)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(token)
    return token


class AccessPolicy:
    def __init__(self, token: str, port: int, extra_hosts: tuple[str, ...] = ()) -> None:
        self.token = token
        names = ["127.0.0.1", "localhost", "[::1]"]
        self.allowed_hosts = {f"{n}:{port}" for n in names} | set(extra_hosts)
        if port == 80:
            self.allowed_hosts |= set(names)
        self.allowed_origins = {f"http://{h}" for h in self.allowed_hosts}

    def host_ok(self, host: str | None) -> bool:
        return host is not None and host in self.allowed_hosts

    def origin_ok(self, origin: str | None) -> bool:
        return origin is None or origin in self.allowed_origins

    def token_ok(self, presented: str | None) -> bool:
        return presented is not None and hmac.compare_digest(presented, self.token)


def _headers(scope: Scope) -> dict[str, str]:
    return {k.decode("latin-1").lower(): v.decode("latin-1") for k, v in scope.get("headers", [])}


class SecurityMiddleware:
    def __init__(self, app: ASGIApp, policy: AccessPolicy) -> None:
        self.app = app
        self.policy = policy

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return
        headers = _headers(scope)
        if not self.policy.host_ok(headers.get("host")):
            await self._reject(scope, send, 421, "host_not_allowed", "Host header not allowed")
            return
        origin = headers.get("origin")
        if scope["type"] == "websocket":
            query = scope.get("query_string", b"").decode("latin-1")
            params = dict(p.split("=", 1) for p in query.split("&") if "=" in p)
            if origin is None or not self.policy.origin_ok(origin):
                await self._reject(scope, send, 403, "origin_not_allowed", "Origin not allowed")
                return
            if not self.policy.token_ok(params.get("token")):
                await self._reject(scope, send, 403, "token_required", "Session token required")
                return
            await self.app(scope, receive, send)
            return
        method = scope.get("method", "GET")
        if not self.policy.origin_ok(origin):
            await self._reject(scope, send, 403, "origin_not_allowed", "Origin not allowed")
            return
        if method not in SAFE_METHODS:
            if not self.policy.token_ok(headers.get(TOKEN_HEADER)):
                await self._reject(scope, send, 403, "token_required", "X-ALife-Token required")
                return
            length = headers.get("content-length")
            if length is not None and (not length.isdigit() or int(length) > MAX_BODY_BYTES):
                await self._reject(scope, send, 413, "body_too_large", "Request body too large")
                return
            receive = _limited_receive(receive)
        await self.app(scope, receive, send)

    async def _reject(self, scope: Scope, send: Send, status: int, code: str, message: str) -> None:
        if scope["type"] == "websocket":
            await send({"type": "websocket.close", "code": 1008, "reason": code})
            return
        body = json.dumps({"api_version": "v1", "error": {"code": code, "message": message}})
        await send(
            {
                "type": "http.response.start",
                "status": status,
                "headers": [(b"content-type", b"application/json")],
            }
        )
        await send({"type": "http.response.body", "body": body.encode("utf-8")})


class BodyTooLarge(Exception):
    pass


def _limited_receive(receive: Receive) -> Receive:
    total = 0

    async def wrapped() -> Message:
        nonlocal total
        message = await receive()
        if message["type"] == "http.request":
            total += len(message.get("body", b""))
            if total > MAX_BODY_BYTES:
                raise BodyTooLarge
        return message

    return wrapped
