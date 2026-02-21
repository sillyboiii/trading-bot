"""
SMA Channel Market Structure
============================
Two SMAs form a price channel:
  Upper SMA: SMA(length) of candle highs
  Lower SMA: SMA(length) of candle lows

Trend rules:
  Bullish : close > upper SMA  → look for LONGS only
  Bearish : close < lower SMA  → look for SHORTS only
  Ranging : close inside channel → no new trades

Entry signals (evaluated per closed candle):
  Breakout Long   : bullish trend + close > recent swing high
  Pullback Long   : bullish trend + price was in channel recently
                    + now closed back above upper SMA
  Breakout Short  : bearish trend + close < recent swing low
  Pullback Short  : bearish trend + price was in channel recently
                    + now closed back below lower SMA
"""

from dataclasses import dataclass
from typing import Optional

import pandas as pd


@dataclass
class SwingPoint:
    index: int
    timestamp: int
    price: float
    kind: str  # 'high' | 'low'


@dataclass
class StructureState:
    trend: str = "ranging"          # 'bullish' | 'bearish' | 'ranging'
    upper_sma: float = 0.0          # SMA(length) of highs
    lower_sma: float = 0.0          # SMA(length) of lows
    recent_swing_high: Optional[SwingPoint] = None
    recent_swing_low: Optional[SwingPoint] = None
    breakout_long: bool = False
    breakout_short: bool = False
    pullback_long: bool = False
    pullback_short: bool = False

    @property
    def has_long_signal(self) -> bool:
        return self.breakout_long or self.pullback_long

    @property
    def has_short_signal(self) -> bool:
        return self.breakout_short or self.pullback_short

    # Compatibility aliases used by risk_manager and backtester
    @property
    def last_valid_high(self) -> Optional[SwingPoint]:
        return self.recent_swing_high

    @property
    def last_valid_low(self) -> Optional[SwingPoint]:
        return self.recent_swing_low


class MarketStructure:
    """
    SMA channel-based trend and signal detector.
    """

    def __init__(
        self,
        sma_length: int = 20,
        pivot_lookback: int = 10,
        trend_confirm_candles: int = 3,
    ):
        self.sma_length = sma_length
        self.pivot_lookback = pivot_lookback
        # How many consecutive candles must be outside the channel before
        # we consider the trend established enough to trade.
        self.trend_confirm_candles = trend_confirm_candles

    # ──────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────

    def analyse(self, df: pd.DataFrame) -> StructureState:
        """
        Analyse a closed-candle OHLCV DataFrame and return the current
        trend + any active entry signals.

        Args:
            df: DataFrame with columns [timestamp, open, high, low, close, volume],
                sorted oldest → newest.

        Returns:
            StructureState
        """
        df = df.reset_index(drop=True)
        state = StructureState()

        min_rows = self.sma_length + self.pivot_lookback + 2
        if len(df) < min_rows:
            return state

        upper_sma = df["high"].rolling(self.sma_length).mean()
        lower_sma = df["low"].rolling(self.sma_length).mean()

        current_close = float(df["close"].iloc[-1])
        state.upper_sma = float(upper_sma.iloc[-1])
        state.lower_sma = float(lower_sma.iloc[-1])

        # ── Step 1: Trend ──────────────────────────────────────
        if current_close > state.upper_sma:
            state.trend = "bullish"
        elif current_close < state.lower_sma:
            state.trend = "bearish"
        else:
            state.trend = "ranging"

        # ── Swing points (SL reference + breakout detection) ───
        state.recent_swing_high = self._find_recent_swing(df, "high")
        state.recent_swing_low = self._find_recent_swing(df, "low")

        # ── Trend confirmation: skip signals until trend is established ───────
        # Count how many of the last N candles (excluding current) were outside
        # the channel in the trend direction.
        n_confirm = self._count_trend_candles(df, upper_sma, lower_sma, state.trend)
        if n_confirm < self.trend_confirm_candles:
            return state  # trend too fresh — no signal yet

        # ── Entry signals (edge-triggered — fire only on the crossover candle) ──
        prev_close = float(df["close"].iloc[-2]) if len(df) >= 2 else current_close
        current_open = float(df["open"].iloc[-1])

        if state.trend == "bullish":
            # Breakout long: THIS candle is the first close above recent swing high.
            # Quality gate: the swing high must sit within 0.5% of the upper SMA —
            # this ensures we're breaking out of a channel-level consolidation, not
            # chasing a trend that has already moved far above the channel.
            if (
                state.recent_swing_high
                and prev_close <= state.recent_swing_high.price
                and current_close > state.recent_swing_high.price
                and state.recent_swing_high.price <= state.upper_sma * 1.005
            ):
                state.breakout_long = True

            # Pullback long: previous candle was inside/below channel,
            # current candle is the first close back above upper SMA.
            # Candle must be bullish (close > open) to confirm rejection momentum.
            prev_upper = float(upper_sma.iloc[-2]) if len(upper_sma) >= 2 else state.upper_sma
            if (
                prev_close <= prev_upper
                and current_close > state.upper_sma
                and current_close > current_open  # bullish close = conviction
            ):
                state.pullback_long = True

        elif state.trend == "bearish":
            # Breakout short: THIS candle is the first close below recent swing low.
            # Quality gate: swing low must be within 0.5% of lower SMA.
            if (
                state.recent_swing_low
                and prev_close >= state.recent_swing_low.price
                and current_close < state.recent_swing_low.price
                and state.recent_swing_low.price >= state.lower_sma * 0.995
            ):
                state.breakout_short = True

            # Pullback short: previous candle was inside/above channel,
            # current candle is the first close back below lower SMA.
            # Candle must be bearish (close < open) to confirm rejection momentum.
            prev_lower = float(lower_sma.iloc[-2]) if len(lower_sma) >= 2 else state.lower_sma
            if (
                prev_close >= prev_lower
                and current_close < state.lower_sma
                and current_close < current_open  # bearish close = conviction
            ):
                state.pullback_short = True

        return state

    # ──────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────

    def _count_trend_candles(
        self,
        df: pd.DataFrame,
        upper_sma: pd.Series,
        lower_sma: pd.Series,
        trend: str,
    ) -> int:
        """
        Count how many of the last 10 prior candles (excluding current) had
        their close outside the channel in the given direction.
        Uses a non-consecutive count so a single pullback candle doesn't
        reset the entire confirmation.
        """
        if trend == "ranging":
            return 0

        lookback = min(10, len(df) - 1)
        count = 0
        for i in range(len(df) - 2, len(df) - 2 - lookback, -1):
            if i < 0:
                break
            c = float(df["close"].iloc[i])
            u = float(upper_sma.iloc[i])
            lo = float(lower_sma.iloc[i])
            if trend == "bullish" and c > u:
                count += 1
            elif trend == "bearish" and c < lo:
                count += 1
        return count

    def _find_recent_swing(
        self, df: pd.DataFrame, kind: str
    ) -> Optional[SwingPoint]:
        """
        Return the most recent confirmed pivot high or low, walking backwards
        through the DataFrame and excluding the last `pivot_lookback` candles
        (which cannot yet be confirmed).
        """
        lb = self.pivot_lookback
        col = "high" if kind == "high" else "low"
        agg_fn = max if kind == "high" else min

        for i in range(len(df) - lb - 1, lb - 1, -1):
            window = df[col].iloc[i - lb: i + lb + 1]
            if float(df[col].iloc[i]) == float(agg_fn(window)):
                return SwingPoint(
                    index=i,
                    timestamp=int(df["timestamp"].iloc[i]),
                    price=float(df[col].iloc[i]),
                    kind=kind,
                )
        return None

