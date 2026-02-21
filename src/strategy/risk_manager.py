"""
Risk Manager
============
SMA Channel strategy risk rules:

Stop Loss:
  Long  → below the lower SMA (with buffer)
  Short → above the upper SMA (with buffer)
  If a recent swing point is tighter than the SMA, use it instead.

Take Profit:
  Primary  → recent swing high (long) / swing low (short)
  Fallback → fixed R:R multiple from entry if no swing point or R:R too low

Minimum R:R: 2.0 : 1
"""

from dataclasses import dataclass
from typing import Optional

from src.strategy.market_structure import SwingPoint


@dataclass
class TradeSetup:
    coin: str
    side: str               # 'long' | 'short'
    entry: float
    stop_loss: float
    take_profit: float
    rr_ratio: float
    position_size_usd: float
    position_size_coin: float
    signal_type: str        # 'breakout' | 'pullback'
    approved: bool          # True only if rr_ratio >= MIN_RR

    def summary(self) -> str:
        status = "✅ APPROVED" if self.approved else "❌ REJECTED (R:R too low)"
        sig = "📈 Breakout" if self.signal_type == "breakout" else "🔄 Pullback"
        return (
            f"{status} | {sig}\n"
            f"  Coin:   {self.coin}\n"
            f"  Side:   {self.side.upper()}\n"
            f"  Entry:  ${self.entry:,.4f}\n"
            f"  SL:     ${self.stop_loss:,.4f}\n"
            f"  TP:     ${self.take_profit:,.4f}\n"
            f"  R:R:    {self.rr_ratio:.2f}:1\n"
            f"  Size:   ${self.position_size_usd:,.2f} "
            f"({self.position_size_coin:.6f} {self.coin})"
        )


class RiskManager:
    def __init__(
        self,
        min_rr: float = 2.0,
        position_size_pct: float = 0.10,
        sl_buffer: float = 0.001,
    ):
        self.min_rr = min_rr
        self.position_size_pct = position_size_pct
        self.sl_buffer = sl_buffer  # e.g., 0.001 = 0.1% beyond SMA

    # ──────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────

    def evaluate(
        self,
        coin: str,
        side: str,
        current_price: float,
        upper_sma: float,
        lower_sma: float,
        recent_swing_high: Optional[SwingPoint],
        recent_swing_low: Optional[SwingPoint],
        account_balance: float,
        signal_type: str = "pullback",
        signal_candle_low: Optional[float] = None,
        signal_candle_high: Optional[float] = None,
    ) -> Optional["TradeSetup"]:
        """
        Build and evaluate a trade setup.

        Returns None if a valid setup cannot be constructed.
        Returns TradeSetup with approved=False if R:R < MIN_RR.
        """
        entry = current_price

        # ── Stop Loss ─────────────────────────────────────────
        if side == "long":
            # Base SL: just below the lower SMA
            sl_sma = lower_sma * (1 - self.sl_buffer)
            # Tighter alternative: just below the signal candle's low.
            # Using the candle's low means the trade is invalidated as soon as
            # the rejection candle structure fails — a much cleaner exit level.
            if signal_candle_low and signal_candle_low > sl_sma:
                sl = signal_candle_low * (1 - self.sl_buffer)
            else:
                sl = sl_sma
        else:  # short
            sl_sma = upper_sma * (1 + self.sl_buffer)
            if signal_candle_high and signal_candle_high < sl_sma:
                sl = signal_candle_high * (1 + self.sl_buffer)
            else:
                sl = sl_sma

        risk = abs(entry - sl)
        if risk <= 0:
            return None

        # ── Take Profit ───────────────────────────────────────
        tp = self._choose_tp(side, entry, sl, recent_swing_high, recent_swing_low)
        if tp is None:
            return None

        # ── R:R ───────────────────────────────────────────────
        rr = self._calc_rr(side, entry, sl, tp)
        if rr <= 0:
            return None

        # ── Position size ─────────────────────────────────────
        pos_usd = account_balance * self.position_size_pct
        pos_coin = pos_usd / current_price if current_price > 0 else 0

        return TradeSetup(
            coin=coin,
            side=side,
            entry=entry,
            stop_loss=round(sl, 6),
            take_profit=round(tp, 6),
            rr_ratio=round(rr, 2),
            position_size_usd=round(pos_usd, 2),
            position_size_coin=round(pos_coin, 6),
            signal_type=signal_type,
            approved=rr >= self.min_rr,
        )

    # ──────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────

    def _choose_tp(
        self,
        side: str,
        entry: float,
        sl: float,
        recent_swing_high: Optional[SwingPoint],
        recent_swing_low: Optional[SwingPoint],
    ) -> Optional[float]:
        """
        Select take profit:
        1. Use the recent swing high (long) / swing low (short) if it meets min_rr.
        2. Fall back to a fixed min_rr multiple.
        """
        risk = abs(entry - sl)
        fixed_tp = (
            entry + self.min_rr * risk
            if side == "long"
            else entry - self.min_rr * risk
        )

        if side == "long":
            if recent_swing_high and recent_swing_high.price > entry:
                swing_rr = (recent_swing_high.price - entry) / risk
                if swing_rr >= self.min_rr:
                    return recent_swing_high.price
            return fixed_tp

        else:  # short
            if recent_swing_low and recent_swing_low.price < entry:
                swing_rr = (entry - recent_swing_low.price) / risk
                if swing_rr >= self.min_rr:
                    return recent_swing_low.price
            return fixed_tp

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
