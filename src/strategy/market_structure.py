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

    def __init__(self, sma_length: int = 20, pivot_lookback: int = 5):
        self.sma_length = sma_length
        self.pivot_lookback = pivot_lookback

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

        # ── Entry signals (edge-triggered — fire only on the crossover candle) ──
        prev_close = float(df["close"].iloc[-2]) if len(df) >= 2 else current_close

        if state.trend == "bullish":
            # Breakout long: THIS candle is the first close above recent swing high
            if (
                state.recent_swing_high
                and prev_close <= state.recent_swing_high.price
                and current_close > state.recent_swing_high.price
            ):
                state.breakout_long = True

            # Pullback long: previous candle was inside/below channel,
            # current candle is the first close back above upper SMA
            prev_upper = float(upper_sma.iloc[-2]) if len(upper_sma) >= 2 else state.upper_sma
            if prev_close <= prev_upper and current_close > state.upper_sma:
                state.pullback_long = True

        elif state.trend == "bearish":
            # Breakout short: THIS candle is the first close below recent swing low
            if (
                state.recent_swing_low
                and prev_close >= state.recent_swing_low.price
                and current_close < state.recent_swing_low.price
            ):
                state.breakout_short = True

            # Pullback short: previous candle was inside/above channel,
            # current candle is the first close back below lower SMA
            prev_lower = float(lower_sma.iloc[-2]) if len(lower_sma) >= 2 else state.lower_sma
            if prev_close >= prev_lower and current_close < state.lower_sma:
                state.pullback_short = True

        return state

    # ──────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────

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

