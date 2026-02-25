"""
Strategy Engine — BOS/CHoCH Supply & Demand Zone Strategy
==========================================================
Multi-timeframe trend-following using supply and demand zones:

  Step 1 → HTF (1h) swing structure trend  (bullish = HH+HL | bearish = LH+LL)
             Primary filter — higher weight. If HTF is ranging, no new trades.

  Step 2 → LTF (5m) swing structure trend must AGREE with HTF.

  Step 3 → Find BOS/CHoCH-validated demand/supply zones on the LTF.
             A zone is only valid if the impulse that formed it broke through
             a confirmed prior swing (BOS or CHoCH).  No BOS/CHoCH = no trade.

  Step 4 → Price retraces into a valid zone → entry signal.
             Side follows the agreed trend (long for bullish, short for bearish).

  Step 5 → Stop Loss just outside zone boundary:
             Long  → below demand zone_low  (− SL_BUFFER)
             Short → above supply zone_high (+ SL_BUFFER)

  Step 6 → Take Profit at next confirmed structure swing. Falls back to
             MIN_RR × risk if no qualifying swing is available.

  Step 7 → Trailing: after breakeven activates, SL trails to each new
             confirmed swing low (long) / swing high (short) on the LTF.

Runs on a timed loop, firing once per closed 5m candle.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from config import Config
from src.exchange.hyperliquid_client import HyperliquidClient
from src.strategy.market_structure import MarketStructure, StructureState
from src.strategy.risk_manager import RiskManager, TradeSetup
from src.strategy.zones import ZoneDetector

logger = logging.getLogger(__name__)


@dataclass
class PairState:
    coin: str
    structure: Optional[StructureState] = None
    htf_trend: str = "ranging"
    last_candle_time: int = 0
    in_trade: bool = False
    trades_taken: int = 0
    active_setup: Optional[TradeSetup] = None
    breakeven_activated: bool = False
    last_ltf_df: Optional[pd.DataFrame] = None   # cached for trailing SL


class StrategyEngine:
    """
    Main engine — evaluates the BOS/CHoCH supply & demand zone strategy
    on every new closed 5m candle.
    """

    def __init__(
        self,
        config: Config,
        client: HyperliquidClient,
        alert_callback=None,
    ):
        self.config = config
        self.client = client
        self.alert_callback = alert_callback

        self.ms_detector = MarketStructure(
            pivot_lookback=config.PIVOT_LOOKBACK,
            atr_length=config.ATR_LENGTH,
        )

        self.zone_detector = ZoneDetector(
            swing_lookback=config.PIVOT_LOOKBACK,
        )

        self._consecutive_losses: int = 0
        self._peak_balance: float = config.PAPER_BALANCE
        self._paused_until: float = 0.0

        self.risk_manager = RiskManager(
            min_rr=config.MIN_RR,
            position_size_pct=config.POSITION_SIZE_PCT,
            sl_buffer=config.SL_BUFFER,
            atr_sl_mult=config.ATR_SL_MULT,
            risk_per_trade_pct=config.RISK_PER_TRADE_PCT,
            max_position_pct=config.MAX_POSITION_PCT,
        )

        self.pair_states: dict[str, PairState] = {
            coin: PairState(coin=coin) for coin in config.PAIRS
        }

        self.running = False
        self._interval_seconds = self._timeframe_to_seconds(config.TIMEFRAME)

    # ──────────────────────────────────────────────────────────
    # Control
    # ──────────────────────────────────────────────────────────

    async def start(self):
        self.running = True
        logger.info(
            f"Strategy engine started | pairs={self.config.PAIRS} "
            f"| ltf={self.config.TIMEFRAME} | htf={self.config.HTF_TIMEFRAME}"
        )
        mode = "🧪 DRY RUN (paper trading)" if self.client.dry_run else (
            "⚠️ TESTNET" if self.config.HL_TESTNET else "🔴 MAINNET"
        )
        await self._alert(
            f"🤖 *Bot started — BOS/CHoCH Zone Strategy*\n"
            f"Mode: {mode}\n"
            f"Pairs: {', '.join(self.config.PAIRS)}\n"
            f"LTF (entry): {self.config.TIMEFRAME}\n"
            f"HTF (trend): {self.config.HTF_TIMEFRAME}\n"
            f"Min R:R: {self.config.MIN_RR}\n"
            f"Risk/trade: {self.config.RISK_PER_TRADE_PCT*100:.1f}%"
        )
        await self._run_loop()

    async def stop(self):
        self.running = False
        logger.info("Strategy engine stopped")
        await self._alert("⛔ *Bot stopped* — no new trades will be placed.")

    # ──────────────────────────────────────────────────────────
    # Main loop
    # ──────────────────────────────────────────────────────────

    async def _run_loop(self):
        while self.running:
            try:
                await self._tick()
            except Exception as e:
                logger.exception(f"Error in strategy loop: {e}")
                await self._alert(f"⚠️ Strategy error: {e}")
            await asyncio.sleep(self._seconds_to_next_candle())

    async def _tick(self):
        if self.client.dry_run:
            await self._check_paper_positions()

        if self._paused_until > time.time():
            remaining = (self._paused_until - time.time()) / 3600
            logger.info(f"Circuit breaker active — {remaining:.1f}h remaining")
            return

        for coin in self.config.PAIRS:
            try:
                await self._process_pair(coin)
            except Exception as e:
                logger.error(f"Error processing {coin}: {e}")

    # ──────────────────────────────────────────────────────────
    # Per-pair logic
    # ──────────────────────────────────────────────────────────

    async def _process_pair(self, coin: str):
        state = self.pair_states[coin]

        # ── Fetch LTF candles (5m) ─────────────────────────────
        df_ltf = self.client.get_candles(coin, self.config.TIMEFRAME, self.config.CANDLE_LIMIT)
        if df_ltf.empty or len(df_ltf) < self.config.PIVOT_LOOKBACK * 3 + 30:
            logger.debug(f"{coin}: insufficient LTF candle data")
            return

        # Drop the in-progress candle — strategy runs on CLOSED candles only
        df_ltf = df_ltf.iloc[:-1].reset_index(drop=True)

        # Dedup: skip if this candle was already processed
        latest_ts = int(df_ltf["timestamp"].iloc[-1])
        if latest_ts == state.last_candle_time:
            return
        state.last_candle_time = latest_ts

        # Cache the LTF df for use by the trailing SL in paper mode
        state.last_ltf_df = df_ltf

        # ── Fetch HTF candles (1h) ─────────────────────────────
        df_htf = self.client.get_candles(
            coin, self.config.HTF_TIMEFRAME, self.config.HTF_CANDLE_LIMIT
        )
        if df_htf.empty or len(df_htf) < self.config.PIVOT_LOOKBACK * 3 + 10:
            logger.debug(f"{coin}: insufficient HTF candle data")
            return
        df_htf = df_htf.iloc[:-1].reset_index(drop=True)

        # ── Step 1: HTF trend (primary filter) ────────────────
        htf_state = self.ms_detector.analyse(df_htf)
        state.htf_trend = htf_state.trend

        if htf_state.trend == "ranging":
            logger.debug(f"{coin}: HTF ranging — skip")
            return

        # ── Step 2: LTF structure ─────────────────────────────
        ltf_state = self.ms_detector.analyse(df_ltf)
        state.structure = ltf_state

        # LTF must agree with HTF — misaligned = no trade
        if ltf_state.trend != "ranging" and ltf_state.trend != htf_state.trend:
            logger.debug(
                f"{coin}: HTF={htf_state.trend} LTF={ltf_state.trend} — misaligned, skip"
            )
            return

        # Use HTF trend as the authoritative direction
        side = "long" if htf_state.trend == "bullish" else "short"
        trend = htf_state.trend

        # ── Step 3: Find BOS/CHoCH-validated zones on LTF ─────
        # find_zones returns only active zones where the impulse caused a BOS/CHoCH
        zones = self.zone_detector.find_zones(df_ltf, trend)

        logger.debug(
            f"{coin} | htf={htf_state.trend} | ltf={ltf_state.trend} "
            f"| zones={len(zones)}"
        )

        # ── Step 4: Check if current price is inside a valid zone ──
        current_price = self.client.get_current_price(coin)
        if not current_price:
            return

        active_zone = self.zone_detector.price_in_zone(current_price, zones)
        if not active_zone:
            return

        # Already in a trade — don't stack
        if self.client.has_open_position(coin):
            logger.debug(f"{coin}: already in trade, skipping zone signal")
            return

        # ── Step 5 + 6: Build trade setup ─────────────────────
        balance = self.client.get_account_balance()

        setup = self.risk_manager.evaluate(
            coin=coin,
            side=side,
            current_price=current_price,
            zone=active_zone,
            recent_swing_high=ltf_state.recent_swing_high,
            recent_swing_low=ltf_state.recent_swing_low,
            account_balance=balance,
            atr=ltf_state.atr,
        )

        if setup is None:
            logger.debug(f"{coin}: could not build trade setup")
            return

        await self._alert(
            f"📊 *{coin} Zone Signal*\n"
            f"  HTF: {htf_state.trend.upper()} | LTF: {ltf_state.trend.upper()}\n"
            f"{setup.summary()}"
        )

        if not setup.approved:
            logger.info(f"{coin}: R:R {setup.rr_ratio} < {self.config.MIN_RR} — rejected")
            return

        # ── Execute ────────────────────────────────────────────
        logger.info(f"Executing {side.upper()} on {coin} | R:R={setup.rr_ratio}")
        success = self.client.enter_trade(
            coin=coin,
            side=side,
            size_coin=setup.position_size_coin,
            stop_loss=setup.stop_loss,
            take_profit=setup.take_profit,
        )

        if success:
            state.in_trade = True
            state.active_setup = setup
            state.breakeven_activated = False
            state.trades_taken += 1
            label = (
                f"🧪 *Paper Trade — {coin}*"
                if self.client.dry_run
                else f"✅ *Trade Entered — {coin}*"
            )
            await self._alert(f"{label}\n{setup.summary()}")
        else:
            await self._alert(f"❌ *Trade execution failed for {coin}*")

    # ──────────────────────────────────────────────────────────
    # Paper position monitoring
    # ──────────────────────────────────────────────────────────

    async def _check_paper_positions(self):
        """
        Each tick:
          1. Breakeven: move SL to entry once BREAKEVEN_AT_R profit is reached.
          2. Trail SL: after breakeven, trail SL to the latest confirmed swing
             low (long) or swing high (short) on the LTF — letting the trade
             run with the trend.
          3. Close: if SL or TP is hit.
          4. Circuit breaker: update on close.
        """
        for coin, state in self.pair_states.items():
            if not state.in_trade or not state.active_setup:
                continue

            price = self.client.get_current_price(coin)
            if not price:
                continue

            setup = state.active_setup
            risk = abs(setup.entry - setup.stop_loss)

            # ── Breakeven ─────────────────────────────────────
            be_r = self.config.BREAKEVEN_AT_R
            if be_r > 0 and not state.breakeven_activated:
                triggered = (
                    (setup.side == "long" and price >= setup.entry + be_r * risk)
                    or
                    (setup.side == "short" and price <= setup.entry - be_r * risk)
                )
                if triggered:
                    setup.stop_loss = setup.entry
                    state.breakeven_activated = True
                    await self._alert(
                        f"🔒 *Breakeven — {coin}*\n"
                        f"  SL moved to entry ${setup.entry:,.4f}"
                    )

            # ── Trail SL after breakeven ──────────────────────
            # Once breakeven is active, trail the SL to each new confirmed
            # swing low (long) / swing high (short) so the trade runs with
            # the trend rather than closing at a fixed TP.
            if state.breakeven_activated and state.last_ltf_df is not None:
                self._trail_sl(state, price)

            # ── SL / TP check ─────────────────────────────────
            outcome: Optional[str] = None
            if setup.side == "long":
                if price <= setup.stop_loss:
                    outcome = "breakeven" if state.breakeven_activated else "loss"
                elif price >= setup.take_profit:
                    outcome = "win"
            else:
                if price >= setup.stop_loss:
                    outcome = "breakeven" if state.breakeven_activated else "loss"
                elif price <= setup.take_profit:
                    outcome = "win"

            if outcome is None:
                continue

            self.client.close_position(coin)
            state.in_trade = False
            state.active_setup = None
            state.breakeven_activated = False

            balance = self.client.get_account_balance()
            if balance > self._peak_balance:
                self._peak_balance = balance

            if outcome == "win":
                self._consecutive_losses = 0
                icon, rr_label = "✅", f"+{setup.rr_ratio:.2f}R"
            elif outcome == "breakeven":
                icon, rr_label = "🔄", "±0.00R (breakeven)"
            else:
                self._consecutive_losses += 1
                icon, rr_label = "❌", "-1.00R"

            await self._alert(
                f"{icon} *Paper Trade Closed — {coin}*\n"
                f"  Side:    {setup.side.upper()}\n"
                f"  Outcome: {outcome.upper()}\n"
                f"  Result:  {rr_label}\n"
                f"  Balance: ${balance:,.2f}"
            )
            logger.info(
                f"[PAPER] {coin} closed — {outcome} | {rr_label} | "
                f"balance=${balance:,.2f}"
            )
            await self._check_circuit_breaker(balance)

    def _trail_sl(self, state: PairState, current_price: float):
        """
        After breakeven is activated, move SL up (long) or down (short) to
        the most recent confirmed swing low / swing high on the LTF.

        This lets the trade run with the trend — SL only ratchets in one
        direction (toward profit), never backwards.
        """
        setup = state.active_setup
        df = state.last_ltf_df
        if df is None or setup is None:
            return

        lb = self.config.PIVOT_LOOKBACK
        col = "low" if setup.side == "long" else "high"
        agg_fn = min if setup.side == "long" else max

        # Find the most recent confirmed swing low (long) or high (short)
        best_swing: Optional[float] = None
        for i in range(lb, len(df) - lb):
            window = df[col].iloc[i - lb: i + lb + 1]
            if float(df[col].iloc[i]) == float(agg_fn(window)):
                best_swing = float(df[col].iloc[i])
                # Keep scanning to find the latest (most recent)

        if best_swing is None:
            return

        buffer = self.config.SL_BUFFER
        if setup.side == "long":
            # Trail SL up to the latest swing low — but only if it's higher than
            # current SL and still below entry (don't go past entry on trail)
            new_sl = best_swing * (1.0 - buffer)
            if new_sl > setup.stop_loss and new_sl < setup.entry:
                logger.debug(
                    f"[TRAIL] {state.coin} long SL {setup.stop_loss:.4f} → {new_sl:.4f}"
                )
                setup.stop_loss = new_sl
        else:
            # Trail SL down to the latest swing high
            new_sl = best_swing * (1.0 + buffer)
            if new_sl < setup.stop_loss and new_sl > setup.entry:
                logger.debug(
                    f"[TRAIL] {state.coin} short SL {setup.stop_loss:.4f} → {new_sl:.4f}"
                )
                setup.stop_loss = new_sl

    # ──────────────────────────────────────────────────────────
    # Circuit breaker
    # ──────────────────────────────────────────────────────────

    async def _check_circuit_breaker(self, balance: float):
        if self._paused_until > time.time():
            return

        reason = None
        limit = self.config.CONSECUTIVE_LOSS_LIMIT
        if limit > 0 and self._consecutive_losses >= limit:
            reason = f"{self._consecutive_losses} consecutive losses"

        dd_pct = self.config.MAX_DAILY_LOSS_PCT
        if dd_pct > 0 and self._peak_balance > 0:
            drawdown = (self._peak_balance - balance) / self._peak_balance
            if drawdown >= dd_pct:
                reason = f"{drawdown*100:.1f}% drawdown from peak ${self._peak_balance:,.2f}"

        if reason:
            pause_secs = self.config.CIRCUIT_PAUSE_HOURS * 3600
            self._paused_until = time.time() + pause_secs
            self._consecutive_losses = 0
            await self._alert(
                f"🛑 *Circuit Breaker Triggered*\n"
                f"  Reason:  {reason}\n"
                f"  Pausing: {self.config.CIRCUIT_PAUSE_HOURS:.0f}h\n"
                f"  Balance: ${balance:,.2f}"
            )
            logger.warning(f"Circuit breaker triggered: {reason}")

    # ──────────────────────────────────────────────────────────
    # Status helpers (used by Telegram bot)
    # ──────────────────────────────────────────────────────────

    def get_status_text(self) -> str:
        lines = ["*BOS/CHoCH Zone Strategy Status*\n"]
        for coin, state in self.pair_states.items():
            ltf_trend = state.structure.trend if state.structure else "unknown"
            htf_trend = state.htf_trend
            trade_str = "IN TRADE" if state.in_trade else "watching"
            lines.append(
                f"*{coin}*: htf=`{htf_trend}` | ltf=`{ltf_trend}` | "
                f"{trade_str} | trades={state.trades_taken}"
            )
        return "\n".join(lines)

    # ──────────────────────────────────────────────────────────
    # Utilities
    # ──────────────────────────────────────────────────────────

    async def _alert(self, msg: str):
        if self.alert_callback:
            try:
                await self.alert_callback(msg)
            except Exception as e:
                logger.error(f"Failed to send alert: {e}")

    def _seconds_to_next_candle(self) -> float:
        now = time.time()
        remainder = now % self._interval_seconds
        return self._interval_seconds - remainder + 2  # +2s buffer

    @staticmethod
    def _timeframe_to_seconds(tf: str) -> int:
        mapping = {
            "1m": 60, "3m": 180, "5m": 300, "15m": 900,
            "30m": 1800, "1h": 3600, "4h": 14400, "1d": 86400,
        }
        return mapping.get(tf, 300)
