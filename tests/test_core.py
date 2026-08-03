"""Tests for the load-bearing logic: book maths, jump detection, exit rules,
the paper account, watchlist selection, and the deployment contract."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest
import sqlalchemy as sa

from snapback.db import utcnow
from snapback.ingest.base import MarketRef, book_metrics
from snapback.ingest.runner import select_watchlist
from snapback.signal import prices

# --- order book -------------------------------------------------------------

def test_book_metrics_picks_best_levels_regardless_of_order():
    bids = [(0.40, 100.0), (0.42, 50.0), (0.38, 10.0)]
    asks = [(0.46, 20.0), (0.44, 80.0)]
    m = book_metrics(bids, asks)
    assert m["bid"] == 0.42
    assert m["ask"] == 0.44
    assert m["mid"] == pytest.approx(0.43)
    assert m["spread"] == pytest.approx(0.02)


def test_book_metrics_depth_counts_only_levels_inside_the_band():
    # mid = 0.50; a level 10 cents away must not inflate tradeable depth.
    bids = [(0.49, 100.0), (0.40, 1000.0)]
    asks = [(0.51, 100.0), (0.60, 1000.0)]
    m = book_metrics(bids, asks, band=0.02)
    assert m["depth_2c"] == pytest.approx(0.49 * 100 + 0.51 * 100, rel=1e-3)


def test_book_metrics_survives_an_empty_side():
    m = book_metrics([(0.30, 10.0)], [])
    assert m["mid"] == 0.30 and m["ask"] is None and m["spread"] is None


def test_book_metrics_empty_book_is_all_none():
    assert book_metrics([], [])["mid"] is None


# --- history storage --------------------------------------------------------

def test_store_deduplicates_bars_sharing_a_timestamp(tmp_path, monkeypatch):
    """Chunked fetches repeat the boundary bar; inserting both crashed a backfill."""
    from snapback import db
    from snapback.backtest.history import HistBar, load_series, store
    from snapback.config import get_settings

    monkeypatch.setenv("SNAPBACK_DB_URL", f"sqlite:///{tmp_path / 'test.db'}")
    get_settings.cache_clear()
    db.get_engine.cache_clear()
    try:
        db.init_db()
        moment = datetime(2026, 7, 1, 12, 0)
        later = datetime(2026, 7, 1, 13, 0)
        written = store(
            "m1",
            [HistBar(ts=moment, mid=0.40), HistBar(ts=moment, mid=0.41), HistBar(ts=later, mid=0.5)],
            60,
        )
        assert written == 2
        # Re-storing overlapping data must be a no-op, not a crash.
        assert store("m1", [HistBar(ts=moment, mid=0.40)], 60) == 0
        assert len(load_series("m1", 60)) == 2
    finally:
        get_settings.cache_clear()
        db.get_engine.cache_clear()


# --- fade simulator ---------------------------------------------------------

def _flat_then_jump(base, *, pre_h=48, jump_to=0.65, revert_to=0.55):
    """Minute series: long flat stretch, a 5-minute spike, then partial reversion."""
    series = [(base + timedelta(minutes=m), 0.50 + 0.001 * (m % 3)) for m in range(0, pre_h * 60, 1)]
    t0 = base + timedelta(hours=pre_h)
    for m in range(5):
        series.append((t0 + timedelta(minutes=m), 0.50 + (jump_to - 0.50) * (m + 1) / 5))
    for m in range(5, 60 * 30):
        series.append((t0 + timedelta(minutes=m), revert_to))
    return series, t0


def test_fade_sim_detects_jump_and_takes_profit_on_reversion():
    from snapback.backtest.fade_sim import simulate_market

    base = datetime(2026, 6, 1)
    series, t0 = _flat_then_jump(base, jump_to=0.65, revert_to=0.52)
    hourly = [(ts, p) for ts, p in series if ts.minute == 0]
    trades = simulate_market("poly:test", "test", series, hourly, close_time=None)
    assert len(trades) == 1
    trade = trades[0]
    assert trade.direction == -1  # faded an up-jump
    assert trade.entry_ts >= t0  # entered after the signal, not on it
    assert trade.exit_reason == "target"
    assert trade.gross_pnl > 0


def test_fade_sim_stops_out_when_the_move_continues():
    from snapback.backtest.fade_sim import simulate_market

    base = datetime(2026, 6, 1)
    series = [(base + timedelta(minutes=m), 0.40 + 0.0005 * (m % 2)) for m in range(48 * 60)]
    t0 = base + timedelta(hours=48)
    # jump up and KEEP going: real news, the fade must lose and stop out
    for m in range(600):
        series.append((t0 + timedelta(minutes=m), min(0.93, 0.40 + 0.001 * m + 0.15)))
    hourly = [(ts, p) for ts, p in series if ts.minute == 0]
    trades = simulate_market("poly:test", "test", series, hourly, close_time=None)
    assert len(trades) == 1
    assert trades[0].exit_reason == "stop"
    assert trades[0].gross_pnl < 0


def test_fade_sim_respects_price_band_and_close_proximity():
    from snapback.backtest.fade_sim import simulate_market

    base = datetime(2026, 6, 1)
    series, _ = _flat_then_jump(base, jump_to=0.97, revert_to=0.94)  # lands in the tail
    hourly = [(ts, p) for ts, p in series if ts.minute == 0]
    assert simulate_market("poly:test", "t", series, hourly, close_time=None) == []

    series2, t02 = _flat_then_jump(base, jump_to=0.65, revert_to=0.52)
    hourly2 = [(ts, p) for ts, p in series2 if ts.minute == 0]
    # market closes 10h after the jump: too close, no trade
    assert (
        simulate_market("poly:test", "t", series2, hourly2, close_time=t02 + timedelta(hours=10))
        == []
    )


def test_fade_sim_one_episode_per_lockout():
    from snapback.backtest.fade_sim import simulate_market

    base = datetime(2026, 6, 1)
    series, t0 = _flat_then_jump(base, jump_to=0.65, revert_to=0.55)
    # a second spike 2h after the first must not open a second trade
    for m in range(120, 125):
        idx = next(i for i, (ts, _) in enumerate(series) if ts >= t0 + timedelta(minutes=m))
        series[idx] = (series[idx][0], 0.70)
    hourly = [(ts, p) for ts, p in series if ts.minute == 0]
    trades = simulate_market("poly:test", "t", series, hourly, close_time=None)
    assert len(trades) == 1


# --- live fade engine -------------------------------------------------------

def _bars(base, prices, *, spread=0.01, depth=5000.0):
    from snapback.signal.engine import Bar

    return [
        Bar(
            ts=base + timedelta(minutes=i),
            mid=p,
            bid=round(p - spread / 2, 4),
            ask=round(p + spread / 2, 4),
            depth=depth,
        )
        for i, p in enumerate(prices)
    ]


def test_engine_fires_on_a_big_instant_spike():
    from snapback.signal.engine import detect_fade_signal

    base = datetime(2026, 7, 1, 9, 0)
    prices = [0.40] * 40 + [0.45, 0.52, 0.58, 0.62, 0.64] + [0.64] * 3
    signal = detect_fade_signal(_bars(base, prices), sigma=0.05)
    assert signal is not None
    assert signal.direction == -1  # fade the up-spike by selling YES
    assert signal.velocity_min <= 5


def test_engine_ignores_a_slow_grind_of_the_same_size():
    """The simulation's core finding: gradual moves are information, not noise."""
    from snapback.signal.engine import detect_fade_signal

    base = datetime(2026, 7, 1, 9, 0)
    # same 0.4 -> 0.64 move, spread across 45 minutes
    prices = [0.40 + 0.24 * (i / 45) for i in range(46)] + [0.64] * 4
    assert detect_fade_signal(_bars(base, prices), sigma=0.05) is None


def test_engine_ignores_a_small_spike():
    from snapback.signal.engine import detect_fade_signal

    base = datetime(2026, 7, 1, 9, 0)
    prices = [0.50] * 40 + [0.52, 0.53, 0.54] + [0.54] * 5
    assert detect_fade_signal(_bars(base, prices), sigma=0.05) is None


def test_engine_rejects_down_spikes_and_cheap_markets():
    """Both calibrated out: down-spikes reverted weakly, sub-30c fades lost."""
    from snapback.signal.engine import _tradeable, detect_fade_signal

    base = datetime(2026, 7, 1, 9, 0)
    far_close = datetime(2026, 12, 1)

    down = [0.60] * 40 + [0.55, 0.48, 0.42, 0.38, 0.36] + [0.36] * 3
    signal = detect_fade_signal(_bars(base, down), sigma=0.05)
    assert signal is not None and signal.jump_logit < 0
    assert _tradeable(signal, far_close) == "down-spike (fades poorly)"

    cheap = [0.08] * 40 + [0.10, 0.14, 0.18, 0.21, 0.22] + [0.22] * 3
    cheap_signal = detect_fade_signal(_bars(base, cheap), sigma=0.05)
    assert cheap_signal is not None
    assert _tradeable(cheap_signal, far_close) == "outside price band"


def test_engine_gates_reject_untradeable_signals():
    from snapback.signal.engine import _tradeable, detect_fade_signal

    base = datetime(2026, 7, 1, 9, 0)
    prices = [0.40] * 40 + [0.45, 0.52, 0.58, 0.62, 0.64] + [0.64] * 3
    far_close = datetime(2026, 12, 1)

    thin = detect_fade_signal(_bars(base, prices, depth=10.0), sigma=0.05)
    assert _tradeable(thin, far_close) == "insufficient depth"

    wide = detect_fade_signal(_bars(base, prices, spread=0.20), sigma=0.05)
    assert _tradeable(wide, far_close) == "spread too wide"

    ok = detect_fade_signal(_bars(base, prices), sigma=0.05)
    assert _tradeable(ok, far_close) is None
    assert _tradeable(ok, base + timedelta(hours=2)) == "too close to resolution"


def test_entry_uses_the_executable_side_of_the_book():
    """Selling YES hits the bid; assuming mid would invent free money."""
    from snapback.signal.engine import _entry_price, _exit_price, detect_fade_signal

    base = datetime(2026, 7, 1, 9, 0)
    prices = [0.40] * 40 + [0.45, 0.52, 0.58, 0.62, 0.64] + [0.64] * 3
    signal = detect_fade_signal(_bars(base, prices, spread=0.02), sigma=0.05)
    assert signal is not None
    assert _entry_price(signal) == signal.bar.bid  # sold into the bid
    assert _entry_price(signal) < signal.bar.mid
    exit_bar = _bars(base, [0.55])[0]
    assert _exit_price(exit_bar, signal.direction) == exit_bar.ask  # bought back at ask


def test_engine_full_loop_opens_and_closes_a_paper_trade(tmp_path, monkeypatch):
    """End-to-end through the database: spike -> short YES at the bid -> revert -> target."""
    from snapback import db
    from snapback.config import get_settings

    monkeypatch.setenv("SNAPBACK_DB_URL", f"sqlite:///{tmp_path / 'engine.db'}")
    get_settings.cache_clear()
    db.get_engine.cache_clear()
    try:
        db.init_db()
        from snapback.signal import engine as fade_engine

        now = db.utcnow().replace(second=0, microsecond=0)
        market_id = "poly:engine-test"
        with db.get_engine().begin() as conn:
            conn.execute(
                db.markets.insert().values(
                    id=market_id,
                    venue="polymarket",
                    venue_id="x",
                    question="Engine test market",
                    watch=True,
                    close_time=now + timedelta(days=30),
                    status="open",
                )
            )
            # 40 flat minutes, then a 5-minute spike 0.40 -> 0.64
            prices = [0.40] * 40 + [0.45, 0.52, 0.58, 0.62, 0.64]
            rows = [
                {
                    "market_id": market_id,
                    "ts": now - timedelta(minutes=len(prices) - i),
                    "mid": p,
                    "bid": round(p - 0.005, 4),
                    "ask": round(p + 0.005, 4),
                    "spread": 0.01,
                    "depth_2c": 5000.0,
                }
                for i, p in enumerate(prices)
            ]
            conn.execute(db.market_bars.insert(), rows)

        result = fade_engine.tick()
        # A 0.55-logit spike clears both rule sets, and each keeps its own
        # book, so one signal is expected to produce one position per strategy.
        assert result["opened"] == 2

        with db.get_engine().connect() as conn:
            trades = conn.execute(
                sa.select(db.paper_trades).order_by(db.paper_trades.c.strategy)
            ).all()
        assert {t.strategy for t in trades} == {"fade", "fade_wide"}
        for trade in trades:
            assert trade.side == "sell_yes"  # faded the up-spike
            assert trade.entry_price == pytest.approx(0.635)  # hit the bid, not the mid

        # Price retraces 75% of the spike: the target should trigger.
        with db.get_engine().begin() as conn:
            conn.execute(
                db.market_bars.insert(),
                [
                    {
                        "market_id": market_id,
                        "ts": now + timedelta(minutes=i),
                        "mid": 0.44,
                        "bid": 0.435,
                        "ask": 0.445,
                        "spread": 0.01,
                        "depth_2c": 5000.0,
                    }
                    for i in (1, 2)
                ],
            )
        closed = fade_engine.tick()
        assert closed["closed"] == 2

        with db.get_engine().connect() as conn:
            trade = conn.execute(sa.select(db.paper_trades)).first()
        assert trade.status == "closed_target"
        assert trade.exit_price == pytest.approx(0.445)  # bought back at the ask
        assert trade.pnl > 0
    finally:
        get_settings.cache_clear()
        db.get_engine.cache_clear()


# --- watchlist selection ----------------------------------------------------

def _ref(market_id, *, days_to_close=30.0, event="e1", oi=1000.0, bid=0.4, ask=0.42):
    return MarketRef(
        id=market_id,
        venue="kalshi",
        venue_id=market_id,
        question=market_id,
        close_time=utcnow() + timedelta(days=days_to_close),
        meta={"open_interest": oi, "volume_24h": 0.0, "yes_bid": bid, "yes_ask": ask,
              "event_ticker": event},
    )


def test_watchlist_respects_the_global_limit():
    refs = [_ref(f"m{i}", event=f"e{i}") for i in range(50)]
    assert len(select_watchlist(refs, limit=10)) == 10


def test_watchlist_caps_one_strike_ladder():
    """Without this, a single CPI ladder eats the whole polling budget."""
    refs = [_ref(f"ladder{i}", event="same") for i in range(20)]
    assert len(select_watchlist(refs, limit=60)) == 3


def test_watchlist_excludes_far_dated_and_expiring_markets():
    assert select_watchlist([_ref("far", days_to_close=900)]) == set()
    assert select_watchlist([_ref("soon", days_to_close=0.5)]) == set()


def test_watchlist_excludes_longshot_tails_and_wide_spreads():
    assert select_watchlist([_ref("tail", bid=0.01, ask=0.02)]) == set()
    assert select_watchlist([_ref("wide", bid=0.30, ask=0.55)]) == set()


def test_watchlist_excludes_illiquid_markets():
    assert select_watchlist([_ref("thin", oi=1.0)]) == set()


# --- price maths ------------------------------------------------------------

def _series(*offsets_and_prices, base=datetime(2026, 7, 1)):
    return [(base + timedelta(hours=h), p) for h, p in offsets_and_prices]


def test_stale_quotes_do_not_count_as_observations():
    """Venues emit bars only on activity; carrying one forward for days would
    turn a gap into a fake 'move'."""
    series = _series((0, 0.40), (200, 0.60))
    base = datetime(2026, 7, 1)
    assert prices.value_at(series, base + timedelta(hours=1)) == 0.40
    assert prices.value_at(series, base + timedelta(hours=50)) is None
    assert prices.move(prices.to_logit_series(series), base, 4.0) is None


def test_move_is_measured_when_both_ends_are_fresh():
    series = _series((0, 0.40), (1, 0.45), (4, 0.55))
    delta = prices.move(prices.to_logit_series(series), datetime(2026, 7, 1), 4.0)
    assert delta is not None and delta > 0


def test_coverage_reports_the_share_of_observed_slots():
    dense = _series(*[(h, 0.5) for h in range(8)])
    sparse = _series((0, 0.5), (7, 0.5))
    base = datetime(2026, 7, 1)
    assert prices.coverage(dense, base, 8.0) == 1.0
    assert prices.coverage(sparse, base, 8.0) < 0.5


def test_logit_move_is_scale_aware():
    """5 points near an even market is small; near a resolved one it is huge."""
    middle = abs(prices.logit(0.55) - prices.logit(0.50))
    tail = abs(prices.logit(0.99) - prices.logit(0.94))
    assert tail > 2 * middle


def test_watchlist_prefers_the_more_liquid_market():
    chosen = select_watchlist(
        [_ref("thin", event="a", oi=200.0), _ref("deep", event="b", oi=99999.0)], limit=1
    )
    assert chosen == {"deep"}


# --- deployment contract ----------------------------------------------------

@pytest.mark.parametrize(
    "given,expected",
    [
        # Hosted providers hand out these two schemes; SQLAlchemy maps both to
        # psycopg2, a driver the deployment does not ship. Rewriting them means
        # the connection string can be pasted from the dashboard unedited.
        # TLS is added because psycopg's default silently accepts plaintext.
        ("postgres://u:p@h/db", "postgresql+psycopg://u:p@h/db?sslmode=require"),
        ("postgresql://u:p@h/db", "postgresql+psycopg://u:p@h/db?sslmode=require"),
        # An existing query string is preserved, not clobbered.
        ("postgresql://u:p@h/db?application_name=x",
         "postgresql+psycopg://u:p@h/db?application_name=x&sslmode=require"),
        # An explicit choice - including a deliberate opt-out - is left alone.
        ("postgresql://u:p@h/db?sslmode=verify-full",
         "postgresql+psycopg://u:p@h/db?sslmode=verify-full"),
        ("postgresql+psycopg://u:p@h/db?sslmode=disable",
         "postgresql+psycopg://u:p@h/db?sslmode=disable"),
    ],
)
def test_postgres_urls_are_pinned_to_driver_and_tls(given, expected):
    from snapback.config import Settings

    assert Settings(db_url=given).resolved_db_url() == expected


def test_schema_compiles_for_postgres():
    """The collector writes from CI to Postgres while tests run on SQLite, so
    nothing catches a SQLite-only column type except an explicit compile."""
    from sqlalchemy.dialects import postgresql
    from sqlalchemy.schema import CreateTable

    from snapback.db import metadata

    for table in metadata.sorted_tables:
        CreateTable(table).compile(dialect=postgresql.dialect())


def test_web_url_in_db_setting_names_the_actual_mistake():
    """A provider dashboard URL pasted into the DB setting must not surface as
    an opaque SQLAlchemy dialect-plugin error four frames down."""
    from snapback.config import Settings

    with pytest.raises(ValueError, match="not a database connection string"):
        Settings(db_url="https://console.neon.tech/app/projects/abc").resolved_db_url()


def test_store_quotes_batches_and_skips_duplicates(tmp_path, monkeypatch):
    """Quote storage must survive a repeat of the same timestamp - a bulk
    insert turns what used to be a skipped row into a UNIQUE violation."""
    monkeypatch.setenv("SNAPBACK_DB_URL", f"sqlite:///{tmp_path / 'q.db'}")
    from snapback import config, db

    config.get_settings.cache_clear()
    db.get_engine.cache_clear()
    db.init_db()

    from snapback.ingest.base import Quote
    from snapback.ingest.runner import _store_quotes

    with db.get_engine().begin() as conn:
        for mid in ("poly:a", "poly:b"):
            conn.execute(
                db.markets.insert().values(
                    id=mid, venue="poly", venue_id=mid, question="q", created_at=db.utcnow()
                )
            )

    ts = datetime(2026, 8, 2, 12, 0, 0)
    q = lambda mid: Quote(
        market_id=mid, ts=ts, mid=0.5, bid=0.49, ask=0.51, spread=0.02,
        last=None, volume=None, liquidity=None, depth_2c=100.0,
    )
    # Same market twice inside one batch, then the whole batch replayed.
    assert _store_quotes([q("poly:a"), q("poly:b"), q("poly:a")]) == 2
    assert _store_quotes([q("poly:a"), q("poly:b")]) == 0

    with db.get_engine().connect() as conn:
        n = conn.execute(sa.select(sa.func.count()).select_from(db.market_bars)).scalar()
    assert n == 2

    config.get_settings.cache_clear()
    db.get_engine.cache_clear()


# --- paper account ----------------------------------------------------------

def test_capital_reflects_the_side_taken_not_the_notional():
    """Shorting YES at 0.85 risks 0.15 a contract, not 0.85. Charging the
    notional both ways would overstate a short several times over."""
    from snapback.signal.engine import trade_capital

    assert trade_capital("sell_yes", 0.85, 100) == pytest.approx(15.0)
    assert trade_capital("buy_yes", 0.30, 100) == pytest.approx(30.0)


def test_account_tracks_realised_pnl_and_open_exposure(tmp_path, monkeypatch):
    monkeypatch.setenv("SNAPBACK_DB_URL", f"sqlite:///{tmp_path / 'a.db'}")
    from snapback import config, db

    config.get_settings.cache_clear()
    db.get_engine.cache_clear()
    db.init_db()

    from snapback.signal.engine import STARTING_BALANCE, account

    with db.get_engine().begin() as conn:
        conn.execute(db.markets.insert().values(
            id="poly:m", venue="poly", venue_id="m", question="q", created_at=db.utcnow()))
        # One closed winner, one still open shorting YES at 0.90.
        conn.execute(db.paper_trades.insert().values(
            market_id="poly:m", strategy="fade", side="sell_yes", entry_price=0.80,
            size=100, status="closed_target", pnl=25.0, entry_ts=db.utcnow()))
        conn.execute(db.paper_trades.insert().values(
            market_id="poly:m", strategy="fade", side="sell_yes", entry_price=0.90,
            size=100, status="open", entry_ts=db.utcnow()))

    with db.get_engine().connect() as conn:
        acct = account(conn)

    assert acct["realised_pnl"] == pytest.approx(25.0)
    assert acct["balance"] == pytest.approx(STARTING_BALANCE + 25.0)
    assert acct["committed"] == pytest.approx(10.0)   # 100 * (1 - 0.90)
    assert acct["free"] == pytest.approx(STARTING_BALANCE + 25.0 - 10.0)
    assert acct["open_positions"] == 1

    config.get_settings.cache_clear()
    db.get_engine.cache_clear()


def test_wide_rule_takes_signals_the_calibrated_one_declines(tmp_path, monkeypatch):
    """The whole point of running both: a spike below 0.50 logit, or priced
    outside 0.30-0.90, must be booked by `fade_wide` and by nothing else."""
    monkeypatch.setenv("SNAPBACK_DB_URL", f"sqlite:///{tmp_path / 'w.db'}")
    from snapback import config, db

    config.get_settings.cache_clear()
    db.get_engine.cache_clear()
    db.init_db()
    from snapback.signal import engine as fade_engine

    fade_engine._SIGMA_AT = None
    now = db.utcnow().replace(second=0, microsecond=0)
    market_id = "poly:wide-only"
    with db.get_engine().begin() as conn:
        conn.execute(db.markets.insert().values(
            id=market_id, venue="polymarket", venue_id="x", question="Wide only",
            watch=True, close_time=now + timedelta(days=30), status="open"))
        # 0.20 -> 0.28: a 0.42-logit spike, under the calibrated 0.50 floor,
        # and entering at 0.28 is below the calibrated 0.30 band as well.
        prices = [0.20] * 40 + [0.22, 0.24, 0.26, 0.27, 0.28]
        conn.execute(db.market_bars.insert(), [
            {"market_id": market_id, "ts": now - timedelta(minutes=len(prices) - i),
             "mid": p, "bid": round(p - 0.005, 4), "ask": round(p + 0.005, 4),
             "spread": 0.01, "depth_2c": 5000.0}
            for i, p in enumerate(prices)
        ])

    result = fade_engine.tick()
    assert result["opened"] == 1
    with db.get_engine().connect() as conn:
        rows = conn.execute(sa.select(db.paper_trades)).all()
    assert [r.strategy for r in rows] == ["fade_wide"]

    config.get_settings.cache_clear()
    db.get_engine.cache_clear()
    fade_engine._SIGMA_AT = None


def test_health_diagnosis_classifies_without_leaking_the_dsn(monkeypatch):
    """The health route is public, so it may name the failure but never the
    host, user or password that produced it."""
    monkeypatch.setenv(
        "SNAPBACK_DB_URL",
        "postgresql://postgres.abc:hunter2@db.abcdefgh.supabase.co:5432/postgres",
    )
    from snapback import config, db
    from snapback.web.app import _db_diagnosis

    config.get_settings.cache_clear()
    db.get_engine.cache_clear()

    d = _db_diagnosis(OSError("could not translate host name: getaddrinfo failed"))
    assert d["reason"] == "dns-ipv6-only"
    assert "pooler" in d["hint"]

    for exc, reason in (
        (Exception('FATAL: password authentication failed for user "postgres"'), "auth"),
        (Exception("FATAL: (ENOTFOUND) tenant or user not found"), "wrong-tenant"),
        (Exception("connection timed out"), "timeout"),
    ):
        assert _db_diagnosis(exc)["reason"] == reason

    blob = " ".join(str(v) for exc in
                    (OSError("getaddrinfo failed"), Exception("password authentication failed"))
                    for v in _db_diagnosis(exc).values())
    for secret in ("hunter2", "abcdefgh", "postgres.abc"):
        assert secret not in blob

    config.get_settings.cache_clear()
    db.get_engine.cache_clear()


@pytest.mark.parametrize(
    "raw,reason",
    [
        # Every one of these parses as a URL and fails only at connect time,
        # where it looks like a network fault instead of the typo it is.
        ("postgresql://u:p@h.pooler.supabase.com:5432/postgres\n", "whitespace"),
        ("postgresql://u:[YOUR-PASSWORD]@h.pooler.supabase.com:5432/postgres", "placeholder"),
        ("postgresql://u:pa@ss@h.pooler.supabase.com:5432/postgres", "unescaped-password"),
        ("postgresql://h.pooler.supabase.com:5432/postgres", "no-credentials"),
        ("u:p@h.pooler.supabase.com:5432/postgres", "no-scheme"),
        ("https://console.supabase.com/project/x", "web-address"),
        # A colon inside the password is legal and must not be flagged.
        ("postgresql://u:pa:ss@h.pooler.supabase.com:5432/postgres", None),
        ("postgresql://u:p@h.pooler.supabase.com:5432/postgres", None),
    ],
)
def test_structural_dsn_faults(raw, reason):
    from snapback.web.app import _structural_fault

    fault = _structural_fault(raw)
    assert (fault or {}).get("reason") == reason
    if fault:
        assert "pa:ss" not in " ".join(fault.values())


# --- watchlist sanitation ---------------------------------------------------

def _poly_ref(mid, bid=None, ask=None, vol24=5000.0, liq=900_000.0, days=60):
    from snapback.ingest.base import MarketRef

    if bid is None and mid is not None:
        bid, ask = mid - 0.005, mid + 0.005
    return MarketRef(
        id=f"poly:{mid}-{vol24}", venue="polymarket", venue_id="v",
        question="q", close_time=utcnow() + timedelta(days=days),
        meta={"best_bid": bid, "best_ask": ask, "volume_24h": vol24,
              "liquidity_num": liq, "volume_num": 12_000_000.0},
    )


def test_a_book_with_no_bid_is_dead_not_unknown():
    """80 of 300 sports outcomes quoted an ask and no bid. Reading that as
    'unknown, allow' is what put the 0.1c F1 drivers on the watchlist."""
    from snapback.ingest.runner import _is_tradeable_band

    assert _is_tradeable_band(_poly_ref(None, bid=None, ask=0.001)) is False
    assert _is_tradeable_band(_poly_ref(0.50)) is True
    # A venue that carries no book at discovery is still genuinely unknown.
    from snapback.ingest.base import MarketRef

    kalshi = MarketRef(id="kalshi:x", venue="kalshi", venue_id="v", question="q",
                       close_time=utcnow() + timedelta(days=30), meta={})
    assert _is_tradeable_band(kalshi) is True


def test_liquidity_score_ignores_the_event_level_figure():
    """`liquidity_num` runs backwards inside a grouped market - the dead
    outcomes carry the largest values - so ranking must not use it."""
    from snapback.ingest.runner import _liquidity_score

    contender = _poly_ref(0.30, vol24=8000.0, liq=240_000.0)
    dead = _poly_ref(0.001, bid=0.001, ask=0.002, vol24=2.86, liq=881_000.0)
    assert _liquidity_score(contender) > _liquidity_score(dead)


def test_fade_band_excludes_markets_the_engine_could_never_enter():
    from snapback.ingest.runner import _in_fade_band

    assert _in_fade_band(_poly_ref(0.50)) is True
    assert _in_fade_band(_poly_ref(0.12)) is True      # can drift into 0.15
    assert _in_fade_band(_poly_ref(0.02)) is False     # would have to septuple
    assert _in_fade_band(_poly_ref(0.99)) is False
    assert _in_fade_band(_poly_ref(None, bid=None, ask=0.001)) is False


def test_legacy_env_prefix_survives_an_empty_new_one(monkeypatch):
    """CI sets a missing secret to "" rather than leaving it unset, so the
    bridge has to treat empty as absent or the rename strands the collector."""
    monkeypatch.setenv("SNAPBACK_DB_URL", "")
    monkeypatch.setenv("PREDGRAPH_DB_URL", "postgresql://u:p@h.pooler.supabase.com:5432/db")
    from snapback import config

    config._adopt_legacy_env()
    config.get_settings.cache_clear()
    assert config.get_settings().db_url.startswith("postgresql://u:p@")
    config.get_settings.cache_clear()
