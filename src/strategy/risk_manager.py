"""
Risk Manager
============
Zone-based Stop Loss (new strategy):
  Long  → SL placed just BELOW the demand zone_low  (entry - SL_BUFFER × zone_low)
  Short → SL placed just ABOVE the supply zone_high (entry + SL_BUFFER × zone_high)

  The zone boundary IS the invalidation level.  If price closes below a demand
  zone it's no longer a valid demand zone, so SL sits right there.

Take Profit (trailing with trend):
  Primary  → next confirmed swing high (long) / swing low (short) if R:R ≥ MIN_RR
  Fallback → fixed MIN_RR multiple from entry

Position Sizing (risk-based, unchanged):
  Dollar risk per trade = account_balance × RISK_PER_TRADE_PCT  (default 1%)
  Position size (coin)  = dollar_risk / (entry − stop_loss)
  Capped at MAX_POSITION_PCT × balance.
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
    signal_type: str        # 'zone_demand' | 'zone_supply'
    approved: bool          # True only if rr_ratio >= MIN_RR
    zone_high: float = 0.0  # zone boundaries stored for trailing reference
    zone_low: float = 0.0

    def summary(self) -> str:
        status = "✅ APPROVED" if self.approved else "❌ REJECTED (R:R too low)"
        icon = "🟢 Demand Zone" if self.signal_type == "zone_demand" else "🔴 Supply Zone"
        return (
            f"{status} | {icon}\n"
            f"  Coin:   {self.coin}\n"
            f"  Side:   {self.side.upper()}\n"
            f"  Entry:  ${self.entry:,.4f}\n"
            f"  Zone:   ${self.zone_low:,.4f} – ${self.zone_high:,.4f}\n"
            f"  SL:     ${self.stop_loss:,.4f}\n"
            f"  TP:     ${self.take_profit:,.4f}\n"
            f"  R:R:    {self.rr_ratio:.2f}:1\n"
            f"  Size:   ${self.position_size_usd:,.2f} "
            f"({self.position_size_coin:.6f} {self.coin})"
        )


class RiskManager:
    def __init__(
        self,
        min_rr: float = 2.5,
        position_size_pct: float = 0.10,     # legacy — only used if risk_per_trade_pct=0
        sl_buffer: float = 0.001,
        atr_sl_mult: float = 1.5,            # kept for ATR fallback when no zone
        risk_per_trade_pct: float = 0.01,
        max_position_pct: float = 0.25,
    ):
        self.min_rr = min_rr
        self.position_size_pct = position_size_pct
        self.sl_buffer = sl_buffer
        self.atr_sl_mult = atr_sl_mult
        self.risk_per_trade_pct = risk_per_trade_pct
        self.max_position_pct = max_position_pct

    # ──────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────

    def evaluate(
        self,
        coin: str,
        side: str,
        current_price: float,
        recent_swing_high: Optional[SwingPoint],
        recent_swing_low: Optional[SwingPoint],
        account_balance: float,
        atr: float = 0.0,
        # Zone-based SL (preferred)
        zone=None,                          # Optional[Zone]
        # Legacy fallback params (used by backtester when no zone)
        upper_sma: float = 0.0,
        lower_sma: float = 0.0,
        signal_type: str = "zone_demand",
        signal_candle_low: Optional[float] = None,
        signal_candle_high: Optional[float] = None,
    ) -> Optional["TradeSetup"]:
        """
        Build and evaluate a trade setup.

        When a zone is provided (live strategy), SL is placed at the zone
        boundary.  Without a zone (backtester fallback), ATR-based SL is used.

        Returns None if a valid setup cannot be constructed.
        Returns TradeSetup with approved=False if R:R < MIN_RR.
        """
        entry = current_price

        # ── Stop Loss ─────────────────────────────────────────
        if zone is not None:
            sl = self._calc_sl_from_zone(side, zone)
            zone_high = zone.zone_high
            zone_low = zone.zone_low
            sig_type = "zone_demand" if side == "long" else "zone_supply"
        else:
            # Fallback: ATR-based (used by backtester)
            sl = self._calc_sl_atr_fallback(
                side, entry, upper_sma, lower_sma,
                signal_candle_low, signal_candle_high, atr
            )
            zone_high = upper_sma
            zone_low = lower_sma
            sig_type = signal_type

        if sl is None:
            return None

        risk = abs(entry - sl)
        if risk <= 0:
            return None

        # ── Take Profit ───────────────────────────────────────
        tp = self._choose_tp(side, entry, sl, recent_swing_high, recent_swing_low)
        if tp is None:
            return None

        rr = self._calc_rr(side, entry, sl, tp)
        if rr <= 0:
            return None

        pos_usd, pos_coin = self._calc_position(account_balance, entry, risk)

        return TradeSetup(
            coin=coin,
            side=side,
            entry=entry,
            stop_loss=round(sl, 6),
            take_profit=round(tp, 6),
            rr_ratio=round(rr, 2),
            position_size_usd=round(pos_usd, 2),
            position_size_coin=round(pos_coin, 6),
            signal_type=sig_type,
            approved=rr >= self.min_rr,
            zone_high=round(zone_high, 6),
            zone_low=round(zone_low, 6),
        )

    # ──────────────────────────────────────────────────────────
    # Stop loss
    # ──────────────────────────────────────────────────────────

    def _calc_sl_from_zone(self, side: str, zone) -> float:
        """
        Place SL just outside the zone boundary with a small buffer.
          Long  (demand zone): SL = zone_low  × (1 − sl_buffer)
          Short (supply zone): SL = zone_high × (1 + sl_buffer)
        """
        if side == "long":
            return zone.zone_low * (1.0 - self.sl_buffer)
        else:
            return zone.zone_high * (1.0 + self.sl_buffer)

    def _calc_sl_atr_fallback(
        self,
        side: str,
        entry: float,
        upper_sma: float,
        lower_sma: float,
        signal_candle_low: Optional[float],
        signal_candle_high: Optional[float],
        atr: float,
    ) -> Optional[float]:
        """ATR-based SL used as a fallback when no zone is available (backtester)."""
        if side == "long":
            atr_sl = (entry - self.atr_sl_mult * atr) if atr > 0 else None
            sma_sl = lower_sma * (1 - self.sl_buffer) if lower_sma > 0 else None
            candle_sl = signal_candle_low * (1 - self.sl_buffer) if signal_candle_low else None

            if atr_sl is not None and atr_sl < entry:
                return atr_sl

            candidates = [c for c in [sma_sl, candle_sl] if c is not None and c < entry]
            return max(candidates) if candidates else None
        else:
            atr_sl = (entry + self.atr_sl_mult * atr) if atr > 0 else None
            sma_sl = upper_sma * (1 + self.sl_buffer) if upper_sma > 0 else None
            candle_sl = signal_candle_high * (1 + self.sl_buffer) if signal_candle_high else None

            if atr_sl is not None and atr_sl > entry:
                return atr_sl

            candidates = [c for c in [sma_sl, candle_sl] if c is not None and c > entry]
            return min(candidates) if candidates else None

    # ──────────────────────────────────────────────────────────
    # Take profit
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
        Take profit target:
          1. Next confirmed swing high (long) / swing low (short) if R:R ≥ MIN_RR.
             This lets the trade run to a natural structure level.
          2. Fixed MIN_RR multiple from entry as fallback.
        """
        risk = abs(entry - sl)
        fixed_tp = (
            entry + self.min_rr * risk if side == "long"
            else entry - self.min_rr * risk
        )

        if side == "long":
            if recent_swing_high and recent_swing_high.price > entry:
                swing_rr = (recent_swing_high.price - entry) / risk
                if swing_rr >= self.min_rr:
                    return recent_swing_high.price
            return fixed_tp
        else:
            if recent_swing_low and recent_swing_low.price < entry:
                swing_rr = (entry - recent_swing_low.price) / risk
                if swing_rr >= self.min_rr:
                    return recent_swing_low.price
            return fixed_tp

    # ──────────────────────────────────────────────────────────
    # Position sizing
    # ──────────────────────────────────────────────────────────

    def _calc_position(
        self,
        account_balance: float,
        entry: float,
        sl_distance: float,
    ) -> tuple[float, float]:
        """
        Risk exactly RISK_PER_TRADE_PCT of balance per trade.
        Capped at MAX_POSITION_PCT × balance.
        """
        dollar_risk = account_balance * self.risk_per_trade_pct
        if sl_distance <= 0 or entry <= 0:
            pos_usd = account_balance * self.position_size_pct
            return pos_usd, pos_usd / entry

        pos_coin = dollar_risk / sl_distance
        pos_usd = pos_coin * entry

        max_usd = account_balance * self.max_position_pct
        if pos_usd > max_usd:
            pos_usd = max_usd
            pos_coin = max_usd / entry

        return pos_usd, pos_coin

    # ──────────────────────────────────────────────────────────
    # R:R
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
