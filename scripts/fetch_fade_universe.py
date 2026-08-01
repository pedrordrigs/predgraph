"""Fetch a category-diverse Polymarket universe for the fade simulation.

Per category tag: top markets by volume (open or recently closed), 90 days of
hourly history, jump detection on the hourly series, then 1-minute windows
around each deduplicated jump. Metadata lands in data/fade_universe.json;
bars land in history_bars (no graph pollution — these markets are a study
universe, not watchlist members).
"""

from __future__ import annotations

import json
import logging
import pathlib
import sys
from datetime import UTC, datetime, timedelta

from predgraph.backtest import history
from predgraph.backtest.fade_sim import JUMP_MIN_ABS_LOGIT
from predgraph.ingest.polymarket import GAMMA, PolymarketClient
from predgraph.signal.damage import detect_jumps

logging.basicConfig(level=logging.WARNING)
log = logging.getLogger("fetch_universe")

CATEGORIES = {
    "sports": "1",
    "crypto": "21",
    "us-politics": "789",
    "geopolitics": "100265",
    "pop-culture": "596",
    "business": "107",
    "tech": "1401",
}
PER_CATEGORY = 45
DAYS = 90
PAD_BEFORE_H = 3
PAD_AFTER_H = 50

OUT = pathlib.Path(__file__).resolve().parent.parent / "data" / "fade_universe.json"


def pick_markets(client: PolymarketClient) -> dict[str, dict]:
    chosen: dict[str, dict] = {}
    for category, tag_id in CATEGORIES.items():
        got = 0
        for closed in ("false", "true"):
            if got >= PER_CATEGORY:
                break
            response = client._client.get(
                f"{GAMMA}/markets",
                params={
                    "tag_id": tag_id,
                    "closed": closed,
                    "active": "true",
                    "limit": 100,
                    "order": "volumeNum",
                    "ascending": "false",
                },
            )
            response.raise_for_status()
            for raw in response.json():
                ref = client._to_ref(raw)
                if ref is None or ref.id in chosen:
                    continue
                # Enough life to have history, and real volume.
                if (ref.meta.get("volume_num") or 0) < 10_000:
                    continue
                if ref.open_time is None:
                    continue
                chosen[ref.id] = {
                    "id": ref.id,
                    "venue": "polymarket",
                    "venue_id": ref.venue_id,
                    "token_id": ref.token_id,
                    "question": ref.question,
                    "category": category,
                    "open_time": ref.open_time.isoformat(),
                    "close_time": ref.close_time.isoformat() if ref.close_time else None,
                    "meta": {"series": None},
                }
                got += 1
                if got >= PER_CATEGORY:
                    break
    return chosen


def main() -> None:
    client = PolymarketClient()
    fetcher = history.HistoryFetcher()
    end = datetime.now(UTC).replace(tzinfo=None)
    start = end - timedelta(days=DAYS)

    try:
        universe = pick_markets(client)
        print(f"universe: {len(universe)} markets", flush=True)
        OUT.write_text(json.dumps(universe, indent=1), encoding="utf-8")

        total_minute_windows = 0
        for n, (market_id, info) in enumerate(universe.items(), 1):
            market = {
                "id": market_id,
                "venue": "polymarket",
                "venue_id": info["venue_id"],
                "token_id": info["token_id"],
                "open_time": datetime.fromisoformat(info["open_time"]),
                "close_time": (
                    datetime.fromisoformat(info["close_time"]) if info["close_time"] else None
                ),
                "meta": {},
            }
            hourly = fetcher.fetch(market, start, end, 60)
            history.store(market_id, hourly, 60)
            series = [(bar.ts, bar.mid) for bar in hourly if bar.mid is not None]
            jumps = detect_jumps(series, z_threshold=3.0, min_abs_logit=JUMP_MIN_ABS_LOGIT)
            for jump in jumps:
                window_start = jump.ts - timedelta(hours=PAD_BEFORE_H)
                window_end = jump.ts + timedelta(hours=PAD_AFTER_H)
                bars = fetcher.fetch(market, window_start, window_end, 1)
                history.store(market_id, bars, 1)
                total_minute_windows += 1
            if n % 25 == 0:
                print(f"  {n}/{len(universe)} markets, {total_minute_windows} minute windows", flush=True)
        print(f"UNIVERSE FETCH DONE: {len(universe)} markets, {total_minute_windows} windows", flush=True)
    finally:
        fetcher.close()
        client.close()


if __name__ == "__main__":
    sys.exit(main())
