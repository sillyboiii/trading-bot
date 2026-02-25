"""
Backtester — BOS/CHoCH Supply & Demand Zone Strategy
======================================================
Walk-forward simulation of the full two-timeframe strategy on historical candles.

At each closed candle the engine:
  1. Resamples LTF (5m) data to HTF (1h) to determine primary trend.
  2. Determines LTF swing structure — must agree with HTF.
  3. Finds BOS/CHoCH-validated demand/supply zones on LTF.
  4. Checks if the closed candle's close is inside a valid zone → entry signal.
  5. Simulates entry at the next candle's open with zone-based SL.

Performance metrics:
  - Total trades taken / rejected
  - Win rate  (price reached TP before SL)
  - Average realised R:R
  - Net return (% of simulated account)
  - Equity curve (balance after each trade)

No actual orders are placed — simulation only.
"""

import io
import logging
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd

from config import Config
from src.exchange.hyperliquid_client import HyperliquidClient
from src.strategy.market_structure import MarketStructure
from src.strategy.risk_manager import RiskManager
from src.strategy.zones import ZoneDetector

logger = logging.getLogger(__name__)

SIM_BALANCE = 10_000.0

_CANDLES_PER_DAY = {
    "1m": 1440, "3m": 480, "5m": 288, "15m": 96,
    "30m": 48, "1h": 24, "4h": 6, "1d": 1,
}

# Minimum 5m candles before we start scanning.
# HTF (1h) needs pivot_lookback*3+5 ≈ 35 hours ≈ 420 candles.
_MIN_5M_START = 420


def _candle_limit(days: int, interval: str) -> int:
    """Convert calendar days to a candle count (minimum 500 for zone detection)."""
    return max(500, days * _CANDLES_PER_DAY.get(interval, 96))


class Backtester:
    def __init__(self, client: HyperliquidClient, config: Config):
        self.client = client
        self.config = config

        self.ms = MarketStructure(
            pivot_lookback=config.PIVOT_LOOKBACK,
            atr_length=config.ATR_LENGTH,
        )
        self.zone_detector = ZoneDetector(
            swing_lookback=config.PIVOT_LOOKBACK,
        )
        self.risk = RiskManager(
            min_rr=config.MIN_RR,
            position_size_pct=config.POSITION_SIZE_PCT,
            sl_buffer=config.SL_BUFFER,
            atr_sl_mult=config.ATR_SL_MULT,
            risk_per_trade_pct=config.RISK_PER_TRADE_PCT,
            max_position_pct=config.MAX_POSITION_PCT,
        )

    # ──────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────

    def run(self, days: int = 30) -> dict:
        """
        Run backtest for all configured pairs on the LTF timeframe.

        Returns dict keyed by coin with result dicts (including 'equity_curve').
        """
        results = {}
        timeframes = [self.config.TIMEFRAME]

        for coin in self.config.PAIRS:
            results[coin] = {}
            for tf in timeframes:
                limit = _candle_limit(days, tf)
                logger.info(f"Backtesting {coin} on {tf} ({days}d → {limit} candles)…")
                df = self.client.get_candles(coin, tf, limit)
                if df.empty:
                    logger.warning(f"No data for {coin} {tf}")
                    continue
                res = self._backtest_pair(coin, df)
                results[coin][tf] = res
                logger.info(
                    f"{coin} {tf}: trades={res['total_trades']} "
                    f"({res['wins']}W/{res['losses']}L/{res['timeouts']}T) "
                    f"winrate={res['win_rate']:.1f}% "
                    f"win_avg_rr={res['avg_win_rr']:.2f} "
                    f"EV={res['ev_per_trade']:+.3f}R "
                    f"net={res['net_pct']:+.1f}%"
                )

        return results

    # ──────────────────────────────────────────────────────────
    # Core simulation
    # ──────────────────────────────────────────────────────────

    def _backtest_pair(self, coin: str, df: pd.DataFrame) -> dict:
        """
        Walk-forward simulation: at each candle i, re-run strategy on all
        prior data, check for a zone entry signal, then simulate the outcome.

        Signal condition: close of candle (i-1) is inside a BOS/CHoCH-validated
        zone, and both HTF and LTF trends agree.
        Entry: open of candle i (next bar open).
        """
        total_trades = 0
        rejected = 0
        wins = 0
        losses = 0
        timeouts = 0
        breakevens = 0
        win_rr_values: list[float] = []
        net_pct = 0.0
        balance = SIM_BALANCE
        equity_curve: list[float] = [SIM_BALANCE]
        in_trade_until: Optional[int] = None

        for i in range(_MIN_5M_START, len(df) - 1):
            if in_trade_until is not None and i <= in_trade_until:
                continue

            # All data up to and including candle i-1 (closed)
            window = df.iloc[:i].copy().reset_index(drop=True)

            # ── HTF trend from resampled LTF data ────────────
            df_htf = self._resample_to_htf(window)
            if len(df_htf) < self.ms.pivot_lookback * 3 + 5:
                continue
            htf_state = self.ms.analyse(df_htf)
            if htf_state.trend == "ranging":
                continue

            # ── LTF swing structure ───────────────────────────
            ltf_state = self.ms.analyse(window)
            if ltf_state.trend != "ranging" and ltf_state.trend != htf_state.trend:
                continue

            trend = htf_state.trend
            side = "long" if trend == "bullish" else "short"

            # ── BOS/CHoCH-validated zones ─────────────────────
            zones = self.zone_detector.find_zones(window, trend)
            if not zones:
                continue

            # Signal candle = last candle of window (df[i-1])
            signal_close = float(df["close"].iloc[i - 1])
            active_zone = self.zone_detector.price_in_zone(signal_close, zones)
            if not active_zone:
                continue

            # Entry at open of the next candle
            entry_price = float(df["open"].iloc[i])

            # Re-check zone validity at entry price (small gaps are OK)
            if not active_zone.contains(entry_price):
                # Allow entry if open is within 0.5% of the zone
                gap_pct = abs(entry_price - active_zone.midpoint) / active_zone.midpoint
                if gap_pct > 0.005:
                    continue

            # ── Build setup ───────────────────────────────────
            setup = self.risk.evaluate(
                coin=coin,
                side=side,
                current_price=entry_price,
                zone=active_zone,
                recent_swing_high=ltf_state.recent_swing_high,
                recent_swing_low=ltf_state.recent_swing_low,
                account_balance=balance,
                atr=ltf_state.atr,
            )

            if setup is None:
                continue

            if not setup.approved:
                rejected += 1
                continue

            total_trades += 1

            # ── Simulate outcome ──────────────────────────────
            outcome = self._simulate_outcome(df, i + 1, setup)

            r = self.config.RISK_PER_TRADE_PCT
            if outcome == "win":
                wins += 1
                win_rr_values.append(setup.rr_ratio)
                net_pct += r * setup.rr_ratio * 100
                balance *= (1 + r * setup.rr_ratio)
            elif outcome == "loss":
                losses += 1
                net_pct -= r * 100
                balance *= (1 - r)
            elif outcome == "breakeven":
                breakevens += 1
            else:
                timeouts += 1

            equity_curve.append(balance)
            in_trade_until = i + 20

        win_rate = (wins / total_trades * 100) if total_trades > 0 else 0.0
        avg_win_rr = (sum(win_rr_values) / len(win_rr_values)) if win_rr_values else 0.0
        loss_rate = losses / total_trades if total_trades > 0 else 0.0
        ev_per_trade = (win_rate / 100) * avg_win_rr - loss_rate

        return {
            "total_trades": total_trades,
            "wins": wins,
            "losses": losses,
            "timeouts": timeouts,
            "breakevens": breakevens,
            "rejected": rejected,
            "win_rate": round(win_rate, 1),
            "avg_win_rr": round(avg_win_rr, 2),
            "ev_per_trade": round(ev_per_trade, 3),
            "net_pct": round(net_pct, 2),
            "final_balance": round(balance, 2),
            "equity_curve": equity_curve,
        }

    def _simulate_outcome(
        self,
        df: pd.DataFrame,
        start_idx: int,
        setup,
        max_candles: int = 50,
    ) -> str:
        """
        Walk forward from start_idx and return 'win', 'loss', 'breakeven', or 'timeout'.
        Mirrors the live engine's breakeven logic.
        """
        be_r = self.config.BREAKEVEN_AT_R
        risk = abs(setup.entry - setup.stop_loss)
        effective_sl = setup.stop_loss
        breakeven_activated = False

        for i in range(start_idx, min(start_idx + max_candles, len(df))):
            high = float(df["high"].iloc[i])
            low = float(df["low"].iloc[i])

            if be_r > 0 and not breakeven_activated:
                if setup.side == "long" and high >= setup.entry + be_r * risk:
                    effective_sl = setup.entry
                    breakeven_activated = True
                elif setup.side == "short" and low <= setup.entry - be_r * risk:
                    effective_sl = setup.entry
                    breakeven_activated = True

            if setup.side == "long":
                if low <= effective_sl:
                    return "breakeven" if breakeven_activated else "loss"
                if high >= setup.take_profit:
                    return "win"
            else:
                if high >= effective_sl:
                    return "breakeven" if breakeven_activated else "loss"
                if low <= setup.take_profit:
                    return "win"

        return "timeout"

    # ──────────────────────────────────────────────────────────
    # HTF resampling
    # ──────────────────────────────────────────────────────────

    def _resample_to_htf(self, df_5m: pd.DataFrame) -> pd.DataFrame:
        """
        Resample 5m OHLCV data to the configured HTF (default 1h).
        Used to determine the primary trend without a separate API call.
        """
        if df_5m.empty:
            return pd.DataFrame()

        try:
            df = df_5m.copy()
            df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
            df = df.set_index("datetime")

            rule = self._tf_to_resample_rule(self.config.HTF_TIMEFRAME)
            df_htf = df.resample(rule).agg(
                timestamp=("timestamp", "first"),
                open=("open", "first"),
                high=("high", "max"),
                low=("low", "min"),
                close=("close", "last"),
                volume=("volume", "sum"),
            ).dropna(subset=["open"]).reset_index(drop=True)

            return df_htf
        except Exception as e:
            logger.debug(f"HTF resample failed: {e}")
            return pd.DataFrame()

    @staticmethod
    def _tf_to_resample_rule(tf: str) -> str:
        mapping = {
            "15m": "15min", "30m": "30min", "1h": "1h",
            "4h": "4h", "1d": "1D",
        }
        return mapping.get(tf, "1h")

    # ──────────────────────────────────────────────────────────
    # Chart generation (unchanged)
    # ──────────────────────────────────────────────────────────

    @staticmethod
    def generate_chart(results: dict, days: int = 30) -> bytes:
        """
        Generate an equity-curve PNG from backtest results.

        Args:
            results: dict returned by run()  {coin: {tf: result_dict}}
            days:    used for the chart title

        Returns:
            PNG image as raw bytes.
        """
        coins = [c for c in results if results[c]]
        n = len(coins)
        if n == 0:
            fig, ax = plt.subplots(figsize=(6, 2))
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            buf = io.BytesIO()
            fig.savefig(buf, format="png")
            plt.close(fig)
            return buf.getvalue()

        fig, axes = plt.subplots(
            n, 1,
            figsize=(10, 4 * n),
            squeeze=False,
            facecolor="#0d1117",
        )
        fig.suptitle(
            f"Equity Curve — {days}-day Backtest  (start ${SIM_BALANCE:,.0f})",
            color="white",
            fontsize=13,
            fontweight="bold",
            y=1.01 if n > 1 else 1.04,
        )

        tf_colors = {"5m": "#58a6ff", "15m": "#f78166"}

        for row, coin in enumerate(coins):
            ax = axes[row][0]
            ax.set_facecolor("#161b22")
            ax.tick_params(colors="white")
            ax.yaxis.label.set_color("white")
            ax.xaxis.label.set_color("white")
            ax.title.set_color("white")
            for spine in ax.spines.values():
                spine.set_edgecolor("#30363d")

            ax.axhline(SIM_BALANCE, color="#6e7681", linewidth=0.8, linestyle="--", label="Start")

            for tf, res in results[coin].items():
                curve = res.get("equity_curve", [])
                if not curve:
                    continue
                color = tf_colors.get(tf, "#e3b341")
                net = res["net_pct"]
                label = (
                    f"{tf}  {net:+.1f}%  "
                    f"({res['wins']}W/{res['losses']}L  "
                    f"WR={res['win_rate']:.0f}%)"
                )
                ax.plot(curve, color=color, linewidth=1.8, label=label)

                xs = range(len(curve))
                ax.fill_between(
                    xs, SIM_BALANCE, curve,
                    where=[v >= SIM_BALANCE for v in curve],
                    alpha=0.12, color="#3fb950",
                )
                ax.fill_between(
                    xs, SIM_BALANCE, curve,
                    where=[v < SIM_BALANCE for v in curve],
                    alpha=0.12, color="#f85149",
                )

            ax.set_title(coin, color="white", fontsize=11)
            ax.set_xlabel("Trade #", color="#8b949e", fontsize=9)
            ax.set_ylabel("Balance (USD)", color="#8b949e", fontsize=9)
            ax.yaxis.set_major_formatter(mticker.FuncFormatter(
                lambda x, _: f"${x:,.0f}"
            ))
            ax.legend(
                facecolor="#21262d", edgecolor="#30363d",
                labelcolor="white", fontsize=8,
            )
            ax.grid(color="#21262d", linewidth=0.5)

        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=140, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        buf.seek(0)
        return buf.read()
