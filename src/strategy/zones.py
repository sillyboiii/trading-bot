"""
Supply & Demand Zone Detection
===============================
Strategy rules:
  Demand zone (uptrend):
    - Identify consolidation followed by a STRONG BULLISH impulse
    - Zone = the candle immediately BEFORE the impulse (zone_low → zone_high)

  Supply zone (downtrend):
    - Identify consolidation followed by a STRONG BEARISH impulse
    - Zone = the candle immediately BEFORE the impulse (zone_low → zone_high)

Zones are invalidated when price closes through them.
Zones are only detected in the direction of the current trend.
"""

from dataclasses import dataclass
from typing import Optional
import pandas as pd
import numpy as np


@dataclass
class Zone:
    kind: str           # 'demand' | 'supply'
    zone_high: float
    zone_low: float
    formed_at_idx: int  # index of the candle that is the zone
    impulse_idx: int    # index of the impulse candle that created this zone
    timestamp: int      # timestamp of zone candle
    active: bool = True

    @property
    def midpoint(self) -> float:
        return (self.zone_high + self.zone_low) / 2

    def contains(self, price: float) -> bool:
        """True if price is inside the zone."""
        return self.zone_low <= price <= self.zone_high

    def is_invalidated_by(self, candle_close: float) -> bool:
        """
        Demand zone: invalidated if price closes BELOW zone_low.
        Supply zone: invalidated if price closes ABOVE zone_high.
        """
        if self.kind == "demand":
            return candle_close < self.zone_low
        else:
            return candle_close > self.zone_high


class ZoneDetector:
    """
    Scans OHLCV data for supply and demand zones.
    Only detects zones aligned with the current trend.
    """

    def __init__(
        self,
        impulse_body_ratio: float = 0.6,
        impulse_size_multiplier: float = 1.5,
        lookback_for_avg: int = 20,
    ):
        self.impulse_body_ratio = impulse_body_ratio
        self.impulse_size_multiplier = impulse_size_multiplier
        self.lookback_for_avg = lookback_for_avg

    # ──────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────

    def find_zones(self, df: pd.DataFrame, trend: str) -> list[Zone]:
        """
        Find all active supply/demand zones in df for the given trend.

        Args:
            df:    OHLCV DataFrame sorted oldest → newest.
            trend: 'bullish' | 'bearish' | 'ranging'

        Returns:
            List of Zone objects (active, not yet invalidated).
        """
        if trend == "ranging":
            return []

        zones: list[Zone] = []
        impulse_indices = self._find_impulses(df, trend)

        for imp_idx in impulse_indices:
            if imp_idx < 1:
                continue

            zone_candle_idx = imp_idx - 1
            zone_high = float(df["high"].iloc[zone_candle_idx])
            zone_low = float(df["low"].iloc[zone_candle_idx])

            kind = "demand" if trend == "bullish" else "supply"

            zone = Zone(
                kind=kind,
                zone_high=zone_high,
                zone_low=zone_low,
                formed_at_idx=zone_candle_idx,
                impulse_idx=imp_idx,
                timestamp=int(df["timestamp"].iloc[zone_candle_idx]),
            )

            # Invalidate if price has already closed through it since formation
            zone = self._check_invalidation(df, zone)
            if zone.active:
                zones.append(zone)

        return zones

    def get_active_zones(self, df: pd.DataFrame, trend: str) -> list[Zone]:
        """
        Returns zones sorted by distance to current price (closest first).
        Only zones price has NOT yet visited since their formation.
        """
        all_zones = self.find_zones(df, trend)
        current_price = float(df["close"].iloc[-1])

        # Keep zones that price hasn't already passed through
        relevant: list[Zone] = []
        for zone in all_zones:
            if trend == "bullish":
                # Demand zone should be below current price (retracement target)
                if zone.zone_high < current_price:
                    relevant.append(zone)
            else:
                # Supply zone should be above current price
                if zone.zone_low > current_price:
                    relevant.append(zone)

        # Sort by proximity to current price
        relevant.sort(key=lambda z: abs(z.midpoint - current_price))
        return relevant

    # ──────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────

    def _is_impulse(
        self,
        df: pd.DataFrame,
        idx: int,
        direction: str,
        avg_body: float,
    ) -> bool:
        """
        Determine if candle at `idx` is a strong impulse in `direction`.

        Criteria:
          1. Body-to-range ratio >= impulse_body_ratio
          2. Body size >= impulse_size_multiplier * avg_body
          3. Candle closes in the right direction
        """
        o = float(df["open"].iloc[idx])
        h = float(df["high"].iloc[idx])
        l = float(df["low"].iloc[idx])
        c = float(df["close"].iloc[idx])

        body = abs(c - o)
        candle_range = h - l

        if candle_range == 0:
            return False

        body_ratio = body / candle_range
        if body_ratio < self.impulse_body_ratio:
            return False

        if body < self.impulse_size_multiplier * avg_body:
            return False

        if direction == "bullish" and c <= o:
            return False
        if direction == "bearish" and c >= o:
            return False

        return True

    def _find_impulses(self, df: pd.DataFrame, trend: str) -> list[int]:
        """Return indices of impulse candles."""
        direction = "bullish" if trend == "bullish" else "bearish"
        impulse_indices = []
        n = len(df)
        lb = self.lookback_for_avg

        bodies = (df["close"] - df["open"]).abs()

        for i in range(lb, n):
            avg_body = float(bodies.iloc[max(0, i - lb) : i].mean())
            if avg_body == 0:
                continue
            if self._is_impulse(df, i, direction, avg_body):
                impulse_indices.append(i)

        return impulse_indices

    def _check_invalidation(self, df: pd.DataFrame, zone: Zone) -> Zone:
        """
        Walk candles after zone formation and check if any close
        invalidated the zone.
        """
        for i in range(zone.impulse_idx + 1, len(df)):
            close = float(df["close"].iloc[i])
            if zone.is_invalidated_by(close):
                zone.active = False
                break
        return zone

    def price_in_zone(self, price: float, zones: list[Zone]) -> Optional[Zone]:
        """Return the first zone that contains `price`, or None."""
        for zone in zones:
            if zone.contains(price):
                return zone
        return None
