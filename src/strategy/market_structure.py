"""
Market Structure Detection
==========================
Implements the strategy's core logic:
  - Valid High: confirmed when price breaks BELOW the previous valid low
  - Valid Low:  confirmed when price breaks ABOVE the previous (swing) high

Trend rules:
  - Bullish: last validated structure point is a low AND price hasn't broken it
  - Bearish: last validated structure point is a high AND price hasn't broken it
"""

from dataclasses import dataclass, field
from typing import Optional
import pandas as pd


@dataclass
class SwingPoint:
    index: int
    timestamp: int
    price: float
    kind: str  # 'high' or 'low'


@dataclass
class StructureState:
    trend: str = "ranging"          # 'bullish' | 'bearish' | 'ranging'
    valid_highs: list = field(default_factory=list)   # list[SwingPoint]
    valid_lows: list = field(default_factory=list)    # list[SwingPoint]
    last_valid_high: Optional[SwingPoint] = None
    last_valid_low: Optional[SwingPoint] = None


class MarketStructure:
    """
    Processes a DataFrame of OHLCV candles and returns the current
    market structure (trend + validated swing highs/lows).
    """

    def __init__(self, pivot_lookback: int = 5):
        self.pivot_lookback = pivot_lookback

    # ──────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────

    def analyse(self, df: pd.DataFrame) -> StructureState:
        """
        Main entry point.

        Args:
            df: DataFrame with columns [timestamp, open, high, low, close, volume]
                sorted oldest → newest.

        Returns:
            StructureState with trend + validated swing points.
        """
        df = df.reset_index(drop=True)
        swing_highs, swing_lows = self._find_swings(df)
        return self._validate_structure(df, swing_highs, swing_lows)

    # ──────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────

    def _find_swings(
        self, df: pd.DataFrame
    ) -> tuple[list[SwingPoint], list[SwingPoint]]:
        """Identify pivot highs and lows using a simple lookback window."""
        lb = self.pivot_lookback
        highs, lows = [], []

        for i in range(lb, len(df) - lb):
            window_h = df["high"].iloc[i - lb : i + lb + 1]
            window_l = df["low"].iloc[i - lb : i + lb + 1]

            if df["high"].iloc[i] == window_h.max():
                highs.append(
                    SwingPoint(
                        index=i,
                        timestamp=int(df["timestamp"].iloc[i]),
                        price=float(df["high"].iloc[i]),
                        kind="high",
                    )
                )
            if df["low"].iloc[i] == window_l.min():
                lows.append(
                    SwingPoint(
                        index=i,
                        timestamp=int(df["timestamp"].iloc[i]),
                        price=float(df["low"].iloc[i]),
                        kind="low",
                    )
                )

        return highs, lows

    def _validate_structure(
        self,
        df: pd.DataFrame,
        swing_highs: list[SwingPoint],
        swing_lows: list[SwingPoint],
    ) -> StructureState:
        """
        Walk through price history and apply the strategy's validation rules:

        A swing LOW becomes VALID when subsequent price breaks ABOVE
        the reference swing HIGH that preceded it.

        A swing HIGH becomes VALID when subsequent price breaks BELOW
        the reference swing LOW that preceded it.

        This is a classic "Break of Structure" (BOS) approach.
        """
        state = StructureState()

        if not swing_highs or not swing_lows:
            return state

        # Build index-keyed lookup for O(1) access
        sh_by_idx = {s.index: s for s in swing_highs}
        sl_by_idx = {s.index: s for s in swing_lows}

        # Merge and sort all swing points chronologically
        all_swings: list[SwingPoint] = sorted(
            swing_highs + swing_lows, key=lambda s: s.index
        )

        valid_highs: list[SwingPoint] = []
        valid_lows: list[SwingPoint] = []

        # Reference levels used for break detection
        # ref_high: the most recent swing high we're watching for a bullish BOS
        # ref_low:  the most recent swing low we're watching for a bearish BOS
        ref_high: Optional[SwingPoint] = None
        ref_low: Optional[SwingPoint] = None

        # Candidates that will be validated once a BOS occurs
        candidate_low: Optional[SwingPoint] = None   # lowest low since last ref_high
        candidate_high: Optional[SwingPoint] = None  # highest high since last ref_low

        # Walk through every candle
        for i in range(len(df)):
            candle_high = float(df["high"].iloc[i])
            candle_low = float(df["low"].iloc[i])

            # ── Update reference & candidate levels when a swing is detected ──
            if i in sh_by_idx:
                swing = sh_by_idx[i]
                # Update ref_high to the latest swing high
                if ref_high is None or swing.price > ref_high.price:
                    ref_high = swing
                # Track the most extreme high as candidate_high
                if candidate_high is None or swing.price > candidate_high.price:
                    candidate_high = swing

            if i in sl_by_idx:
                swing = sl_by_idx[i]
                if ref_low is None or swing.price < ref_low.price:
                    ref_low = swing
                if candidate_low is None or swing.price < candidate_low.price:
                    candidate_low = swing

            # ── Bullish BOS: price breaks ABOVE ref_high ──────────────────────
            # → validates candidate_low (the lowest low between last BOS and now)
            if ref_high and candidate_low and candle_high > ref_high.price:
                # The low that formed before this break is now a VALID LOW
                if candidate_low.index < i:
                    valid_lows.append(candidate_low)
                    state.last_valid_low = candidate_low
                # Reset: the broken high is now the new validated reference
                ref_high = None
                candidate_low = None
                # The current position could become a new candidate_high
                candidate_high = None

            # ── Bearish BOS: price breaks BELOW ref_low ───────────────────────
            # → validates candidate_high (the highest high between last BOS and now)
            if ref_low and candidate_high and candle_low < ref_low.price:
                if candidate_high.index < i:
                    valid_highs.append(candidate_high)
                    state.last_valid_high = candidate_high
                ref_low = None
                candidate_high = None
                candidate_low = None

        state.valid_highs = valid_highs
        state.valid_lows = valid_lows

        # ── Determine trend ────────────────────────────────────────────────────
        state.trend = self._determine_trend(df, state)
        return state

    def _determine_trend(self, df: pd.DataFrame, state: StructureState) -> str:
        """
        Trend rules from the strategy:
          Bullish  → last confirmed structure is a VALID LOW & price hasn't broken it
          Bearish  → last confirmed structure is a VALID HIGH & price hasn't broken it
        """
        current_close = float(df["close"].iloc[-1])

        lvh = state.last_valid_high
        lvl = state.last_valid_low

        if lvh is None and lvl is None:
            return "ranging"

        # Determine which came most recently
        if lvl and lvh:
            if lvl.index > lvh.index:
                # Most recent structure is a low → bullish bias
                if current_close > lvl.price:
                    return "bullish"
                else:
                    return "ranging"   # low broken → structure lost
            else:
                # Most recent structure is a high → bearish bias
                if current_close < lvh.price:
                    return "bearish"
                else:
                    return "ranging"
        elif lvl:
            return "bullish" if current_close > lvl.price else "ranging"
        elif lvh:
            return "bearish" if current_close < lvh.price else "ranging"

        return "ranging"
