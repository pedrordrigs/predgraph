"""HTTP access to venues whose domains the local ISP resolver refuses to answer.

The ISP resolver returns NXDOMAIN for the Polymarket and Kalshi domains, but the
endpoints themselves serve read-only market data normally. So instead of a VPN we
resolve the affected hostnames against public resolvers, dial the returned IP
directly, and keep the real hostname in both the TLS SNI and the Host header —
TLS verification still happens against the real certificate name.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from functools import lru_cache

import dns.resolver
import httpx

from predgraph.config import get_settings

log = logging.getLogger(__name__)

MIN_TTL_SECONDS = 60
MAX_TTL_SECONDS = 3600
USER_AGENT = "predgraph/0.1 (research)"


@dataclass(slots=True)
class _CacheEntry:
    ips: list[str]
    expires_at: float


class PinnedResolver:
    """Resolves selected hostnames against public DNS servers, with a TTL cache."""

    def __init__(
        self,
        servers: list[str],
        suffixes: list[str],
        timeout: float = 5.0,
    ) -> None:
        self._suffixes = tuple(s.lower().lstrip(".") for s in suffixes)
        self._cache: dict[str, _CacheEntry] = {}
        self._lock = threading.Lock()
        self._resolver = dns.resolver.Resolver(configure=False)
        self._resolver.nameservers = list(servers)
        self._resolver.timeout = timeout
        self._resolver.lifetime = timeout

    def handles(self, host: str | None) -> bool:
        if not host:
            return False
        h = host.lower()
        return any(h == suffix or h.endswith("." + suffix) for suffix in self._suffixes)

    def resolve(self, host: str) -> list[str]:
        now = time.monotonic()
        with self._lock:
            cached = self._cache.get(host)
            if cached is not None and cached.expires_at > now:
                return cached.ips

        answer = self._resolver.resolve(host, "A")
        ips = [record.address for record in answer]
        if not ips:
            raise httpx.ConnectError(f"no A records for {host}")
        ttl = min(max(int(answer.rrset.ttl), MIN_TTL_SECONDS), MAX_TTL_SECONDS)

        with self._lock:
            self._cache[host] = _CacheEntry(ips, now + ttl)
        log.debug("resolved %s -> %s (ttl %ss)", host, ips, ttl)
        return ips


class PinnedTransport(httpx.HTTPTransport):
    """Dials the resolved IP while presenting the original hostname to TLS."""

    def __init__(self, resolver: PinnedResolver, **kwargs) -> None:
        super().__init__(**kwargs)
        self._resolver = resolver

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        host = request.url.host
        if not self._resolver.handles(host):
            return super().handle_request(request)

        last_error: Exception | None = None
        for ip in self._resolver.resolve(host):
            pinned = httpx.Request(
                method=request.method,
                url=request.url.copy_with(host=ip),
                headers=request.headers,
                stream=request.stream,
                extensions={**request.extensions, "sni_hostname": host},
            )
            # httpx derives Host from the URL, which now holds the IP.
            pinned.headers["Host"] = host
            try:
                return super().handle_request(pinned)
            except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
                log.warning("connect %s via %s failed: %s", host, ip, exc)
                last_error = exc
        raise last_error if last_error else httpx.ConnectError(f"unreachable: {host}")


@lru_cache
def get_resolver() -> PinnedResolver:
    settings = get_settings()
    return PinnedResolver(settings.dns_server_list, settings.dns_suffix_list)


def build_client(
    base_url: str = "",
    *,
    timeout: float | None = None,
    headers: dict[str, str] | None = None,
) -> httpx.Client:
    settings = get_settings()
    transport = PinnedTransport(get_resolver(), retries=1)
    return httpx.Client(
        base_url=base_url,
        transport=transport,
        timeout=timeout if timeout is not None else settings.http_timeout,
        headers={"User-Agent": USER_AGENT, "Accept": "application/json", **(headers or {})},
        follow_redirects=True,
    )
