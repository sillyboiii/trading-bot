"""
Swing-Based Market Structure
============================
Trend is classified using swing structure (Higher Highs/Lows vs Lower Highs/Lows):

  Bullish : most recent swing high > previous swing high  (HH)
            AND most recent swing low  > previous swing low   (HL)

  Bearish : most recent swing high < previous swing high  (LH)
            AND most recent swing low  < previous swing low   (LL)

  Ranging : mixed — no clear HH/HL or LH/LL pattern

Used on two timeframes:
  HTF (1h)  — primary trend filter; higher weight
  LTF (5m)  — entry timeframe; BOS/CHoCH-validated zones (zones.py) required

A confirmed swing is a pivot whose high/low is the highest/lowest within a
±pivot_lookback candle window.  The last pivot_lookback candles are excluded
because they are not yet confirmed.
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
    trend: str = "ranging"              # 'bullish' | 'bearish' | 'ranging'
    atr: float = 0.0
    recent_swing_high: Optional[SwingPoint] = None
    recent_swing_low: Optional[SwingPoint] = None

    # ── Back-compat aliases ────────────────────────────────────
    # These are referenced by risk_manager, backtester, and status helpers.
    upper_sma: float = 0.0      # populated with recent_swing_high.price
    lower_sma: float = 0.0      # populated with recent_swing_low.price
    macro_ema: float = 0.0
    sma_slope: float = 0.0
    breakout_long: bool = False
    breakout_short: bool = False
    pullback_long: bool = False
    pullback_short: bool = False

    @property
    def has_long_signal(self) -> bool:
        return self.pullback_long or self.breakout_long

    @property
    def has_short_signal(self) -> bool:
        return self.pullback_short or self.breakout_short

    @property
    def last_valid_high(self) -> Optional[SwingPoint]:
        return self.recent_swing_high

    @property
    def last_valid_low(self) -> Optional[SwingPoint]:
        return self.recent_swing_low


class MarketStructure:
    """
    Swing-based market structure analyser.

    Determines trend direction from swing highs/lows (HH/HL = bullish,
    LH/LL = bearish, mixed = ranging).  ATR is computed for downstream
    SL sizing fallback.
    """

    def __init__(
        self,
        pivot_lookback: int = 10,
        atr_length: int = 14,
        # ── Legacy params — accepted silently so existing init calls don't break ──
        sma_length: int = 20,
        trend_confirm_candles: int = 5,
        pullback_only: bool = True,
        volume_mult: float = 0.0,
        slope_candles: int = 5,
        macro_ema_period: int = 0,
    ):
        self.pivot_lookback = pivot_lookback
        self.atr_length = atr_length

    # ──────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────

    def analyse(self, df: pd.DataFrame) -> StructureState:
        """
        Analyse a closed-candle OHLCV DataFrame and return the current
        market structure state.

        Args:
            df: DataFrame with columns [timestamp, open, high, low, close, volume],
                sorted oldest → newest.

        Returns:
            StructureState with trend, swing reference points, and ATR.
        """
        df = df.reset_index(drop=True)
        state = StructureState()

        # Need at least 3× lookback to confirm two swing points
        if len(df) < self.pivot_lookback * 3 + 5:
            return state

        state.atr = self._calc_atr(df)

        swing_highs = self._find_all_swings(df, "high")
        swing_lows = self._find_all_swings(df, "low")

        # Need at least two of each to compare structure
        if len(swing_highs) < 2 or len(swing_lows) < 2:
            return state

        state.recent_swing_high = swing_highs[-1]
        state.recent_swing_low = swing_lows[-1]

        # Populate back-compat aliases used by risk_manager / backtester
        state.upper_sma = state.recent_swing_high.price
        state.lower_sma = state.recent_swing_low.price

        state.trend = self._determine_trend(swing_highs, swing_lows)

        return state

    # ──────────────────────────────────────────────────────────
    # Trend classification
    # ──────────────────────────────────────────────────────────

    def _determine_trend(
        self,
        swing_highs: list[SwingPoint],
        swing_lows: list[SwingPoint],
    ) -> str:
        """
        Compare the last two confirmed swing highs and lows.

          HH + HL → 'bullish'
          LH + LL → 'bearish'
          Mixed   → 'ranging'
        """
        hh = swing_highs[-1].price > swing_highs[-2].price   # Higher High
        hl = swing_lows[-1].price > swing_lows[-2].price     # Higher Low
        lh = swing_highs[-1].price < swing_highs[-2].price   # Lower High
        ll = swing_lows[-1].price < swing_lows[-2].price     # Lower Low

        if hh and hl:
            return "bullish"
        if lh and ll:
            return "bearish"
        return "ranging"

    # ──────────────────────────────────────────────────────────
    # Swing point detection
    # ──────────────────────────────────────────────────────────

    def _find_all_swings(self, df: pd.DataFrame, kind: str) -> list[SwingPoint]:
        """
        Find all confirmed pivot highs (kind='high') or lows (kind='low').

        A candle at index i is a confirmed pivot if its high (or low) equals
        the maximum (or minimum) of the window [i-lb … i … i+lb].  The last
        lb candles are excluded because they cannot yet be confirmed.

        Returns a list sorted oldest → newest.
        """
        lb = self.pivot_lookback
        col = "high" if kind == "high" else "low"
        agg_fn = max if kind == "high" else min
        result: list[SwingPoint] = []

        for i in range(lb, len(df) - lb):
            window = df[col].iloc[i - lb: i + lb + 1]
            if float(df[col].iloc[i]) == float(agg_fn(window)):
                result.append(SwingPoint(
                    index=i,
                    timestamp=int(df["timestamp"].iloc[i]),
                    price=float(df[col].iloc[i]),
                    kind=kind,
                ))
        return result

    # ──────────────────────────────────────────────────────────
    # ATR
    # ──────────────────────────────────────────────────────────

    def _calc_atr(self, df: pd.DataFrame) -> float:
        """ATR(atr_length) using Wilder's smoothing (EWM with alpha=1/N)."""
        if len(df) < self.atr_length + 1:
            return 0.0
        high = df["high"]
        low = df["low"]
        prev_close = df["close"].shift(1)
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1 / self.atr_length, min_periods=self.atr_length).mean()
        return float(atr.iloc[-1])

    # ──────────────────────────────────────────────────────────
    # Legacy stubs
    # ──────────────────────────────────────────────────────────

    def _find_recent_swing(self, df: pd.DataFrame, kind: str) -> Optional[SwingPoint]:
        """Return the most recent confirmed swing. Kept for back-compat."""
        swings = self._find_all_swings(df, kind)
        return swings[-1] if swings else None
