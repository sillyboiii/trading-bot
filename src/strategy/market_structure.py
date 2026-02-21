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

Entry signals:
  Pullback Long  : price briefly pulled into the channel, current candle
                   closes back above upper SMA with bullish body + volume.
  Pullback Short : price briefly pulled into channel, current candle
                   closes back below lower SMA with bearish body + volume.
  Breakout Long  : (optional) close breaks above recent swing high near channel.
  Breakout Short : (optional) close breaks below recent swing low near channel.

Additional filters (all must pass before a signal fires):
  - Trend confirmation  : ≥ TREND_CONFIRM_CANDLES of last 10 closes outside channel
  - SMA slope           : channel must be sloping in the trade direction
  - Volume              : signal candle volume ≥ VOLUME_MULT × 20-period avg volume
"""

from dataclasses import dataclass, field
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
    atr: float = 0.0                # ATR(atr_length) — used for SL sizing
    sma_slope: float = 0.0          # fractional slope of the channel midpoint
    macro_ema: float = 0.0          # EMA(macro_ema_period) — 6h macro trend
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
    SMA channel-based trend and signal detector with ATR, slope, and volume filters.
    """

    def __init__(
        self,
        sma_length: int = 20,
        pivot_lookback: int = 10,
        trend_confirm_candles: int = 5,
        atr_length: int = 14,
        pullback_only: bool = True,
        volume_mult: float = 1.2,
        slope_candles: int = 5,
        macro_ema_period: int = 72,
    ):
        self.sma_length = sma_length
        self.pivot_lookback = pivot_lookback
        self.trend_confirm_candles = trend_confirm_candles
        self.atr_length = atr_length
        self.pullback_only = pullback_only
        self.volume_mult = volume_mult
        self.slope_candles = slope_candles
        # 0 = disabled; 72 on 5m ≈ 6h trend filter
        self.macro_ema_period = macro_ema_period

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

        # ── ATR ────────────────────────────────────────────────
        state.atr = self._calc_atr(df)

        # ── Macro EMA ──────────────────────────────────────────
        # EMA(72) on 5m ≈ 6h trend. Signals are only taken in its direction.
        if self.macro_ema_period > 0 and len(df) >= self.macro_ema_period:
            state.macro_ema = float(
                df["close"].ewm(span=self.macro_ema_period, adjust=False).mean().iloc[-1]
            )

        # ── Channel midpoint slope ─────────────────────────────
        # Positive slope = channel moving up (bullish bias), negative = down.
        state.sma_slope = self._calc_slope(upper_sma, lower_sma)

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

        # ── Trend confirmation ─────────────────────────────────
        n_confirm = self._count_trend_candles(df, upper_sma, lower_sma, state.trend)
        if n_confirm < self.trend_confirm_candles:
            return state  # trend too fresh — no signal yet

        # ── Macro EMA filter ───────────────────────────────────
        # Check that the SMA CHANNEL is positioned in the macro trend direction,
        # NOT the current price. During a pullback entry, price has dipped into
        # the channel and current_close may temporarily sit below the EMA even
        # in a genuine uptrend. Using upper_sma/lower_sma instead avoids
        # rejecting valid pullback setups.
        if state.macro_ema > 0:
            if state.trend == "bullish" and state.upper_sma < state.macro_ema:
                return state  # channel below macro trend — skip longs
            if state.trend == "bearish" and state.lower_sma > state.macro_ema:
                return state  # channel above macro trend — skip shorts

        # ── Volume filter ──────────────────────────────────────
        vol_ok = self._check_volume(df)

        # ── Entry signals ──────────────────────────────────────
        prev_close = float(df["close"].iloc[-2]) if len(df) >= 2 else current_close
        current_open = float(df["open"].iloc[-1])

        if state.trend == "bullish":
            # Pullback long: previous candle was inside/below channel,
            # current candle is the first close back above upper SMA.
            # Body must be bullish. Volume must be above average.
            prev_upper = float(upper_sma.iloc[-2]) if len(upper_sma) >= 2 else state.upper_sma
            if (
                prev_close <= prev_upper
                and current_close > state.upper_sma
                and current_close > current_open   # bullish body = conviction
                and vol_ok
            ):
                state.pullback_long = True

            # Breakout long (disabled by default — prone to fakeouts on 5m)
            if not self.pullback_only:
                if (
                    state.recent_swing_high
                    and prev_close <= state.recent_swing_high.price
                    and current_close > state.recent_swing_high.price
                    and state.recent_swing_high.price <= state.upper_sma * 1.005
                    and vol_ok
                ):
                    state.breakout_long = True

        elif state.trend == "bearish":
            # Pullback short: previous candle was inside/above channel,
            # current candle is the first close back below lower SMA.
            # Body must be bearish. Volume must be above average.
            prev_lower = float(lower_sma.iloc[-2]) if len(lower_sma) >= 2 else state.lower_sma
            if (
                prev_close >= prev_lower
                and current_close < state.lower_sma
                and current_close < current_open   # bearish body = conviction
                and vol_ok
            ):
                state.pullback_short = True

            # Breakout short (disabled by default)
            if not self.pullback_only:
                if (
                    state.recent_swing_low
                    and prev_close >= state.recent_swing_low.price
                    and current_close < state.recent_swing_low.price
                    and state.recent_swing_low.price >= state.lower_sma * 0.995
                    and vol_ok
                ):
                    state.breakout_short = True

        return state

    # ──────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────

    def _calc_atr(self, df: pd.DataFrame) -> float:
        """True Range → ATR(atr_length) using Wilder's smoothing (EWM)."""
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
        # Wilder's smoothing: equivalent to EWM with alpha = 1/atr_length
        atr = tr.ewm(alpha=1 / self.atr_length, min_periods=self.atr_length).mean()
        return float(atr.iloc[-1])

    def _calc_slope(self, upper_sma: pd.Series, lower_sma: pd.Series) -> float:
        """
        Fractional slope of the channel midpoint over the last slope_candles bars.
        Positive = rising, negative = falling.
        """
        n = self.slope_candles
        if len(upper_sma) < n + 1:
            return 0.0
        mid_now = (upper_sma.iloc[-1] + lower_sma.iloc[-1]) / 2
        mid_prev = (upper_sma.iloc[-1 - n] + lower_sma.iloc[-1 - n]) / 2
        if mid_prev == 0:
            return 0.0
        return (mid_now - mid_prev) / mid_prev

    def _check_volume(self, df: pd.DataFrame) -> bool:
        """
        Return True if the last candle's volume exceeds volume_mult × 20-period avg.
        Falls back to True if no volume column or insufficient data.
        """
        if "volume" not in df.columns or len(df) < 21:
            return True
        avg_vol = float(df["volume"].iloc[-21:-1].mean())
        if avg_vol <= 0:
            return True
        return float(df["volume"].iloc[-1]) >= self.volume_mult * avg_vol

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
        Non-consecutive — a single pullback candle does not reset the count.
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
