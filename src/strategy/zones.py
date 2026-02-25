"""
Supply & Demand Zone Detection with BOS/CHoCH Validation
=========================================================
Strategy rules:

  Demand zone (bullish trend):
    - A strong bullish impulse candle is detected
    - That impulse MUST have broken through a confirmed prior swing high
      (Break of Structure / Change of Character)
    - Zone = the candle immediately BEFORE the impulse (base)

  Supply zone (bearish trend):
    - A strong bearish impulse candle is detected
    - That impulse MUST have broken through a confirmed prior swing low (BOS/CHoCH)
    - Zone = the candle immediately BEFORE the impulse (base)

  A zone WITHOUT a BOS/CHoCH is invalid and will not be traded.
  Zones are invalidated when price closes through them.
"""

from dataclasses import dataclass
from typing import Optional

import pandas as pd


@dataclass
class Zone:
    kind: str           # 'demand' | 'supply'
    zone_high: float
    zone_low: float
    formed_at_idx: int  # index of the base candle (the zone itself)
    impulse_idx: int    # index of the impulse that created + validated the zone
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
    Scans OHLCV data for supply and demand zones that are:
      1. Created by a strong impulse candle, AND
      2. Validated by a BOS or CHoCH (the impulse broke a prior confirmed swing).

    Only detects zones aligned with the current trend.
    """

    def __init__(
        self,
        impulse_body_ratio: float = 0.6,
        impulse_size_multiplier: float = 1.5,
        lookback_for_avg: int = 20,
        swing_lookback: int = 10,       # pivot window for BOS/CHoCH swing detection
        bos_search_candles: int = 40,   # how far back to search for a swing to break
    ):
        self.impulse_body_ratio = impulse_body_ratio
        self.impulse_size_multiplier = impulse_size_multiplier
        self.lookback_for_avg = lookback_for_avg
        self.swing_lookback = swing_lookback
        self.bos_search_candles = bos_search_candles

    # ──────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────

    def find_zones(self, df: pd.DataFrame, trend: str) -> list[Zone]:
        """
        Find all active supply/demand zones that are BOS/CHoCH-validated.

        Args:
            df:    OHLCV DataFrame sorted oldest → newest.
            trend: 'bullish' | 'bearish' | 'ranging'

        Returns:
            List of Zone objects that are active (not yet invalidated) and
            were formed by an impulse that caused a BOS or CHoCH.
        """
        if trend == "ranging":
            return []

        direction = "bullish" if trend == "bullish" else "bearish"
        zones: list[Zone] = []

        impulse_indices = self._find_impulses(df, trend)

        for imp_idx in impulse_indices:
            if imp_idx < 1:
                continue

            # ── BOS / CHoCH requirement ────────────────────────
            # Skip impulse if it didn't break through a prior swing.
            if not self._is_bos_choch(df, imp_idx, direction):
                continue

            # Zone = base candle immediately before the impulse
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

            zone = self._check_invalidation(df, zone)
            if zone.active:
                zones.append(zone)

        return zones

    def get_active_zones(self, df: pd.DataFrame, trend: str) -> list[Zone]:
        """
        Returns BOS/CHoCH-validated zones sorted by proximity to current price.
        Only zones that price has NOT yet revisited since their formation.
        """
        all_zones = self.find_zones(df, trend)
        current_price = float(df["close"].iloc[-1])

        relevant: list[Zone] = []
        for zone in all_zones:
            if trend == "bullish":
                # Demand zone should be below current price (retracement target)
                if zone.zone_high < current_price:
                    relevant.append(zone)
            else:
                # Supply zone should be above current price (rally target)
                if zone.zone_low > current_price:
                    relevant.append(zone)

        relevant.sort(key=lambda z: abs(z.midpoint - current_price))
        return relevant

    def price_in_zone(self, price: float, zones: list[Zone]) -> Optional[Zone]:
        """Return the first zone that contains `price`, or None."""
        for zone in zones:
            if zone.contains(price):
                return zone
        return None

    # ──────────────────────────────────────────────────────────
    # BOS / CHoCH validation
    # ──────────────────────────────────────────────────────────

    def _is_bos_choch(
        self,
        df: pd.DataFrame,
        imp_idx: int,
        direction: str,
    ) -> bool:
        """
        Return True if the impulse candle at imp_idx caused a BOS or CHoCH.

        Logic:
          Bullish impulse: its close must exceed a confirmed swing HIGH
                           that formed in the preceding bos_search_candles candles.
          Bearish impulse: its close must fall below a confirmed swing LOW
                           in the same lookback window.

        A confirmed swing is a pivot that is the highest/lowest in a
        ±swing_lookback window — the same definition used in market_structure.py.
        """
        lb = self.swing_lookback
        close = float(df["close"].iloc[imp_idx])
        search_start = max(lb, imp_idx - self.bos_search_candles)

        if direction == "bullish":
            # Check if impulse close broke above any prior confirmed swing high
            for i in range(search_start, imp_idx - 1):
                w_start = max(0, i - lb)
                w_end = min(len(df), i + lb + 1)
                if w_end - w_start < 3:
                    continue
                if float(df["high"].iloc[i]) == float(df["high"].iloc[w_start:w_end].max()):
                    # i is a confirmed swing high
                    if close > float(df["high"].iloc[i]):
                        return True   # BOS or CHoCH — impulse broke above this level
        else:
            # Bearish: check if impulse close broke below any prior confirmed swing low
            for i in range(search_start, imp_idx - 1):
                w_start = max(0, i - lb)
                w_end = min(len(df), i + lb + 1)
                if w_end - w_start < 3:
                    continue
                if float(df["low"].iloc[i]) == float(df["low"].iloc[w_start:w_end].min()):
                    # i is a confirmed swing low
                    if close < float(df["low"].iloc[i]):
                        return True   # BOS or CHoCH — impulse broke below this level

        return False

    # ──────────────────────────────────────────────────────────
    # Impulse detection (unchanged from original)
    # ──────────────────────────────────────────────────────────

    def _is_impulse(
        self,
        df: pd.DataFrame,
        idx: int,
        direction: str,
        avg_body: float,
    ) -> bool:
        """
        True if candle at idx is a strong impulse in `direction`.

        Criteria:
          1. Body-to-range ratio >= impulse_body_ratio
          2. Body size >= impulse_size_multiplier × avg_body
          3. Candle closes in the correct direction
        """
        o = float(df["open"].iloc[idx])
        h = float(df["high"].iloc[idx])
        l = float(df["low"].iloc[idx])
        c = float(df["close"].iloc[idx])

        body = abs(c - o)
        candle_range = h - l

        if candle_range == 0:
            return False
        if body / candle_range < self.impulse_body_ratio:
            return False
        if body < self.impulse_size_multiplier * avg_body:
            return False
        if direction == "bullish" and c <= o:
            return False
        if direction == "bearish" and c >= o:
            return False
        return True

    def _find_impulses(self, df: pd.DataFrame, trend: str) -> list[int]:
        """Return indices of all impulse candles matching trend direction."""
        direction = "bullish" if trend == "bullish" else "bearish"
        impulse_indices = []
        lb = self.lookback_for_avg
        bodies = (df["close"] - df["open"]).abs()

        for i in range(lb, len(df)):
            avg_body = float(bodies.iloc[max(0, i - lb): i].mean())
            if avg_body == 0:
                continue
            if self._is_impulse(df, i, direction, avg_body):
                impulse_indices.append(i)

        return impulse_indices

    def _check_invalidation(self, df: pd.DataFrame, zone: Zone) -> Zone:
        """
        Walk candles after zone formation and mark zone inactive if price
        has already closed through it.
        """
        for i in range(zone.impulse_idx + 1, len(df)):
            close = float(df["close"].iloc[i])
            if zone.is_invalidated_by(close):
                zone.active = False
                break
        return zone
