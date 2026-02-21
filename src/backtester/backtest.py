"""
Backtester — SMA Channel Strategy
===================================
Walk-forward simulation of the full strategy on historical candles.

At each closed candle the engine checks:
  1. SMA channel trend (SMA-High / SMA-Low)
  2. Entry signal (breakout or pullback)
  3. R:R filter (>= 2.0 : 1)

Performance metrics reported:
  - Total trades taken / rejected
  - Win rate  (price reached TP before SL)
  - Average realised R:R
  - Net return (% of simulated account)

No actual orders are placed — simulation only.
"""

import logging
from typing import Optional

import pandas as pd

from config import Config
from src.exchange.hyperliquid_client import HyperliquidClient
from src.strategy.market_structure import MarketStructure
from src.strategy.risk_manager import RiskManager

logger = logging.getLogger(__name__)

BACKTEST_CANDLE_LIMIT = 1000
SIM_BALANCE = 10_000.0


class Backtester:
    def __init__(self, client: HyperliquidClient, config: Config):
        self.client = client
        self.config = config

        self.ms = MarketStructure(
            sma_length=config.SMA_LENGTH,
            pivot_lookback=config.PIVOT_LOOKBACK,
        )
        self.risk = RiskManager(
            min_rr=config.MIN_RR,
            position_size_pct=config.POSITION_SIZE_PCT,
            sl_buffer=config.SL_BUFFER,
        )

    # ──────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────

    def run(self) -> dict:
        """
        Run backtest for all configured pairs.
        Compares 5m vs 15m when on either of those timeframes.
        Returns dict keyed by coin with result dicts.
        """
        results = {}
        timeframes = [self.config.TIMEFRAME]

        if self.config.TIMEFRAME in ("5m", "15m"):
            timeframes = ["5m", "15m"]

        for coin in self.config.PAIRS:
            results[coin] = {}
            for tf in timeframes:
                logger.info(f"Backtesting {coin} on {tf}…")
                df = self.client.get_candles(coin, tf, BACKTEST_CANDLE_LIMIT)
                if df.empty:
                    logger.warning(f"No data for {coin} {tf}")
                    continue
                res = self._backtest_pair(coin, df)
                results[coin][tf] = res
                logger.info(
                    f"{coin} {tf}: trades={res['total_trades']} "
                    f"winrate={res['win_rate']:.1f}% "
                    f"avg_rr={res['avg_rr']:.2f} "
                    f"net={res['net_pct']:+.1f}%"
                )

        return results

    # ──────────────────────────────────────────────────────────
    # Core simulation
    # ──────────────────────────────────────────────────────────

    def _backtest_pair(self, coin: str, df: pd.DataFrame) -> dict:
        """
        Walk-forward simulation: at each candle, re-run strategy on all
        prior data, check for an entry signal, then simulate the outcome.
        """
        total_trades = 0
        rejected = 0
        wins = 0
        losses = 0
        rr_values: list[float] = []
        net_pct = 0.0
        balance = SIM_BALANCE

        min_start = self.config.SMA_LENGTH + self.config.PIVOT_LOOKBACK * 2 + 5
        in_trade_until: Optional[int] = None

        for i in range(min_start, len(df) - 1):
            if in_trade_until is not None and i <= in_trade_until:
                continue

            # Data available up to and including candle i-1 (closed)
            window = df.iloc[:i].copy().reset_index(drop=True)

            # Step 1 + 2: trend + signal
            structure = self.ms.analyse(window)

            if structure.trend == "ranging":
                continue

            if structure.trend == "bullish" and structure.has_long_signal:
                side = "long"
                signal_type = "breakout" if structure.breakout_long else "pullback"
            elif structure.trend == "bearish" and structure.has_short_signal:
                side = "short"
                signal_type = "breakout" if structure.breakout_short else "pullback"
            else:
                continue

            # Entry at open of the next candle
            entry_price = float(df["open"].iloc[i])

            # Step 3: R:R
            # The signal candle is the last candle of the analysis window (df[i-1]).
            # Its low/high are used as a tighter SL anchor than the full SMA channel.
            signal_candle_low = float(df["low"].iloc[i - 1])
            signal_candle_high = float(df["high"].iloc[i - 1])
            setup = self.risk.evaluate(
                coin=coin,
                side=side,
                current_price=entry_price,
                upper_sma=structure.upper_sma,
                lower_sma=structure.lower_sma,
                recent_swing_high=structure.recent_swing_high,
                recent_swing_low=structure.recent_swing_low,
                account_balance=balance,
                signal_type=signal_type,
                signal_candle_low=signal_candle_low,
                signal_candle_high=signal_candle_high,
            )

            if setup is None:
                continue

            if not setup.approved:
                rejected += 1
                continue

            total_trades += 1

            # Simulate outcome on subsequent candles
            outcome = self._simulate_outcome(df, i + 1, setup)

            if outcome == "win":
                wins += 1
                rr_values.append(setup.rr_ratio)
                net_pct += self.config.POSITION_SIZE_PCT * setup.rr_ratio * 100
                balance *= (1 + self.config.POSITION_SIZE_PCT * setup.rr_ratio)
            elif outcome == "loss":
                losses += 1
                rr_values.append(-1.0)
                net_pct -= self.config.POSITION_SIZE_PCT * 100
                balance *= (1 - self.config.POSITION_SIZE_PCT)

            in_trade_until = i + 20  # assume max 20 candles per trade

        win_rate = (wins / total_trades * 100) if total_trades > 0 else 0.0
        avg_rr = (sum(rr_values) / len(rr_values)) if rr_values else 0.0

        return {
            "total_trades": total_trades,
            "wins": wins,
            "losses": losses,
            "rejected": rejected,
            "win_rate": round(win_rate, 1),
            "avg_rr": round(avg_rr, 2),
            "net_pct": round(net_pct, 2),
        }

    def _simulate_outcome(
        self,
        df: pd.DataFrame,
        start_idx: int,
        setup,
        max_candles: int = 50,
    ) -> str:
        """
        Walk forward from start_idx and return 'win', 'loss', or 'timeout'.
        Win = price reaches TP before SL.
        """
        for i in range(start_idx, min(start_idx + max_candles, len(df))):
            high = float(df["high"].iloc[i])
            low = float(df["low"].iloc[i])

            if setup.side == "long":
                if low <= setup.stop_loss:
                    return "loss"
                if high >= setup.take_profit:
                    return "win"
            else:
                if high >= setup.stop_loss:
                    return "loss"
                if low <= setup.take_profit:
                    return "win"

        return "timeout"
