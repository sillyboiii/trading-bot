"""
Risk Manager
============
SMA Channel strategy risk rules:

Stop Loss (ATR-based):
  Long  → entry - ATR_SL_MULT × ATR(14)
  Short → entry + ATR_SL_MULT × ATR(14)
  The ATR floor ensures the SL is never hit by normal intraday noise.

Take Profit:
  Primary  → recent swing high (long) / swing low (short) if R:R ≥ MIN_RR
  Fallback → fixed MIN_RR multiple from entry

Position Sizing (risk-based):
  Dollar risk per trade = account_balance × RISK_PER_TRADE_PCT  (default 1%)
  Position size (coin)  = dollar_risk / (entry − stop_loss)
  This means the same fraction of capital is lost on every losing trade
  regardless of ATR, price, or timeframe — the system automatically sizes
  up in low-volatility regimes and down in high-volatility regimes.
  Capped at MAX_POSITION_PCT × balance to prevent over-sizing.

Minimum R:R: 2.5 : 1  (default)
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
        min_rr: float = 2.5,
        position_size_pct: float = 0.10,     # legacy — only used if risk_per_trade_pct=0
        sl_buffer: float = 0.001,
        atr_sl_mult: float = 1.5,
        risk_per_trade_pct: float = 0.01,    # fraction of balance to risk per trade
        max_position_pct: float = 0.25,      # max fraction of balance in one position
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
        upper_sma: float,
        lower_sma: float,
        recent_swing_high: Optional[SwingPoint],
        recent_swing_low: Optional[SwingPoint],
        account_balance: float,
        signal_type: str = "pullback",
        signal_candle_low: Optional[float] = None,
        signal_candle_high: Optional[float] = None,
        atr: float = 0.0,
    ) -> Optional["TradeSetup"]:
        """
        Build and evaluate a trade setup.

        Returns None if a valid setup cannot be constructed.
        Returns TradeSetup with approved=False if R:R < MIN_RR.
        """
        entry = current_price

        # ── Stop Loss (ATR-based) ─────────────────────────────
        sl = self._calc_sl(side, entry, upper_sma, lower_sma,
                           signal_candle_low, signal_candle_high, atr)
        if sl is None:
            return None

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

        # ── Position size (risk-based) ─────────────────────────
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
            signal_type=signal_type,
            approved=rr >= self.min_rr,
        )

    # ──────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────

    def _calc_sl(
        self,
        side: str,
        entry: float,
        upper_sma: float,
        lower_sma: float,
        signal_candle_low: Optional[float],
        signal_candle_high: Optional[float],
        atr: float,
    ) -> Optional[float]:
        """
        Compute stop loss using the tightest of:
          1. ATR-based distance  (entry ± ATR_SL_MULT × ATR)
          2. SMA-based level     (lower_sma for longs, upper_sma for shorts)

        Tighter is safer — it gives a smaller R amount so TP targets are
        proportionally closer too, but avoids the SMA-based SL being hit
        by random noise on short timeframes.
        """
        if side == "long":
            # ATR floor: place SL at entry - atr_sl_mult × ATR
            atr_sl = (entry - self.atr_sl_mult * atr) if atr > 0 else None
            # SMA fallback
            sma_sl = lower_sma * (1 - self.sl_buffer)
            # Signal candle tighter alternative
            candle_sl = (
                signal_candle_low * (1 - self.sl_buffer)
                if signal_candle_low else None
            )

            # Candidates: take the highest (tightest to entry) that is still
            # below entry, and is >= the ATR floor if ATR is available.
            candidates = [c for c in [atr_sl, sma_sl, candle_sl]
                          if c is not None and c < entry]
            if not candidates:
                return None

            if atr_sl is not None:
                # Use ATR level as the SL — it's already volatility-scaled.
                # Ignore candle/SMA alternatives to keep SL consistent.
                return atr_sl
            return max(candidates)  # tightest that is below entry

        else:  # short
            atr_sl = (entry + self.atr_sl_mult * atr) if atr > 0 else None
            sma_sl = upper_sma * (1 + self.sl_buffer)
            candle_sl = (
                signal_candle_high * (1 + self.sl_buffer)
                if signal_candle_high else None
            )
            candidates = [c for c in [atr_sl, sma_sl, candle_sl]
                          if c is not None and c > entry]
            if not candidates:
                return None
            if atr_sl is not None:
                return atr_sl
            return min(candidates)  # tightest above entry

    def _calc_position(
        self,
        account_balance: float,
        entry: float,
        sl_distance: float,
    ) -> tuple[float, float]:
        """
        Risk-based sizing: risk exactly RISK_PER_TRADE_PCT of balance per trade.

        position_coin = dollar_risk / sl_distance
        position_usd  = position_coin × entry
        Capped at MAX_POSITION_PCT × balance.
        """
        dollar_risk = account_balance * self.risk_per_trade_pct
        if sl_distance <= 0 or entry <= 0:
            # Fallback to legacy flat sizing
            pos_usd = account_balance * self.position_size_pct
            return pos_usd, pos_usd / entry

        pos_coin = dollar_risk / sl_distance
        pos_usd = pos_coin * entry

        # Cap at max_position_pct of balance
        max_usd = account_balance * self.max_position_pct
        if pos_usd > max_usd:
            pos_usd = max_usd
            pos_coin = max_usd / entry

        return pos_usd, pos_coin

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
