"""
Risk Manager
============
Handles:
  - Stop loss placement  (just outside the zone)
  - Take profit placement (last valid high/low)
  - R:R ratio calculation
  - Position sizing (% of account)
  - Trade approval gate (R:R >= MIN_RR)
"""

from dataclasses import dataclass
from typing import Optional

from src.strategy.zones import Zone
from src.strategy.market_structure import SwingPoint


@dataclass
class TradeSetup:
    coin: str
    side: str           # 'long' | 'short'
    entry: float
    stop_loss: float
    take_profit: float
    rr_ratio: float
    position_size_usd: float
    position_size_coin: float
    zone: Zone
    approved: bool      # True only if rr_ratio >= MIN_RR

    def summary(self) -> str:
        status = "✅ APPROVED" if self.approved else "❌ REJECTED (R:R too low)"
        return (
            f"{status}\n"
            f"  Coin:     {self.coin}\n"
            f"  Side:     {self.side.upper()}\n"
            f"  Entry:    ${self.entry:,.4f}\n"
            f"  SL:       ${self.stop_loss:,.4f}\n"
            f"  TP:       ${self.take_profit:,.4f}\n"
            f"  R:R:      {self.rr_ratio:.2f}:1\n"
            f"  Size:     ${self.position_size_usd:,.2f} ({self.position_size_coin:.6f} {self.coin})"
        )


class RiskManager:
    def __init__(
        self,
        min_rr: float = 2.5,
        position_size_pct: float = 0.10,
        sl_buffer: float = 0.001,
    ):
        self.min_rr = min_rr
        self.position_size_pct = position_size_pct
        self.sl_buffer = sl_buffer  # e.g., 0.001 = 0.1% beyond zone

    # ──────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────

    def evaluate(
        self,
        coin: str,
        side: str,
        current_price: float,
        zone: Zone,
        last_valid_high: Optional[SwingPoint],
        last_valid_low: Optional[SwingPoint],
        account_balance: float,
    ) -> Optional[TradeSetup]:
        """
        Build and evaluate a trade setup.

        Returns None if take profit cannot be determined (no valid structure).
        Returns TradeSetup with approved=False if R:R < MIN_RR.
        """
        entry = current_price

        if side == "long":
            sl = zone.zone_low * (1 - self.sl_buffer)
            tp = last_valid_high.price if last_valid_high else None
        else:  # short
            sl = zone.zone_high * (1 + self.sl_buffer)
            tp = last_valid_low.price if last_valid_low else None

        if tp is None:
            return None

        rr = self._calc_rr(side, entry, sl, tp)
        if rr <= 0:
            return None

        pos_usd = account_balance * self.position_size_pct
        pos_coin = pos_usd / current_price if current_price > 0 else 0

        return TradeSetup(
            coin=coin,
            side=side,
            entry=entry,
            stop_loss=sl,
            take_profit=tp,
            rr_ratio=round(rr, 2),
            position_size_usd=round(pos_usd, 2),
            position_size_coin=round(pos_coin, 6),
            zone=zone,
            approved=rr >= self.min_rr,
        )

    # ──────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────

    def _calc_rr(self, side: str, entry: float, sl: float, tp: float) -> float:
        if side == "long":
            risk = entry - sl
            reward = tp - entry
        else:
            risk = sl - entry
            reward = entry - tp

        if risk <= 0:
            return 0.0

        return reward / risk
