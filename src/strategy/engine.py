"""
Strategy Engine
================
Orchestrates the 3-step trading strategy for all configured pairs:

  Step 1 → Market structure (trend)
  Step 2 → Supply / demand zones
  Step 3 → R:R filter (≥ 2.5 : 1)

Runs on a timed loop, firing once per closed candle.
Sends Telegram alerts and executes trades via the exchange client.
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
from src.strategy.zones import ZoneDetector, Zone
from src.strategy.risk_manager import RiskManager, TradeSetup

logger = logging.getLogger(__name__)


@dataclass
class PairState:
    coin: str
    structure: Optional[StructureState] = None
    active_zones: list = field(default_factory=list)
    last_candle_time: int = 0
    in_trade: bool = False
    trades_taken: int = 0
    trades_won: int = 0


class StrategyEngine:
    """
    Main engine — runs continuously and evaluates the strategy
    on every new closed candle.
    """

    def __init__(
        self,
        config: Config,
        client: HyperliquidClient,
        alert_callback=None,  # async callable(msg: str)
    ):
        self.config = config
        self.client = client
        self.alert_callback = alert_callback

        self.ms_detector = MarketStructure(pivot_lookback=config.PIVOT_LOOKBACK)
        self.zone_detector = ZoneDetector(
            impulse_body_ratio=config.IMPULSE_BODY_RATIO,
            impulse_size_multiplier=config.IMPULSE_SIZE_MULTIPLIER,
        )
        self.risk_manager = RiskManager(
            min_rr=config.MIN_RR,
            position_size_pct=config.POSITION_SIZE_PCT,
            sl_buffer=config.SL_BUFFER,
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
            f"| tf={self.config.TIMEFRAME} | rr_min={self.config.MIN_RR}"
        )
        await self._alert(
            f"🤖 *Bot started*\n"
            f"Pairs: {', '.join(self.config.PAIRS)}\n"
            f"Timeframe: {self.config.TIMEFRAME}\n"
            f"Min R:R: {self.config.MIN_RR}"
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
        """Evaluate strategy for all pairs on the latest closed candle."""
        for coin in self.config.PAIRS:
            try:
                await self._process_pair(coin)
            except Exception as e:
                logger.error(f"Error processing {coin}: {e}")

    async def _process_pair(self, coin: str):
        state = self.pair_states[coin]

        # ── Fetch candles ──────────────────────────────────────
        df = self.client.get_candles(coin, self.config.TIMEFRAME, self.config.CANDLE_LIMIT)
        if df.empty or len(df) < self.config.PIVOT_LOOKBACK * 2 + 5:
            logger.debug(f"{coin}: insufficient candle data")
            return

        # Work on the CLOSED candle only (drop the last in-progress one)
        df = df.iloc[:-1].reset_index(drop=True)

        # Deduplicate: skip if we already processed this candle
        latest_ts = int(df["timestamp"].iloc[-1])
        if latest_ts == state.last_candle_time:
            return
        state.last_candle_time = latest_ts

        # ── Step 1: Market structure ───────────────────────────
        structure = self.ms_detector.analyse(df)
        state.structure = structure

        logger.debug(
            f"{coin} | trend={structure.trend} "
            f"| valid_highs={len(structure.valid_highs)} "
            f"| valid_lows={len(structure.valid_lows)}"
        )

        if structure.trend == "ranging":
            return  # No trades in ranging markets

        # ── Step 2: Supply / demand zones ─────────────────────
        zones = self.zone_detector.get_active_zones(df, structure.trend)
        state.active_zones = zones

        if not zones:
            logger.debug(f"{coin}: no active zones in trend direction")
            return

        # ── Check if price is in a zone ────────────────────────
        current_price = self.client.get_current_price(coin)
        if not current_price:
            return

        touched_zone = self.zone_detector.price_in_zone(current_price, zones)
        if not touched_zone:
            return

        # Skip if already in a trade for this coin
        if state.in_trade or self.client.has_open_position(coin):
            logger.debug(f"{coin}: already in trade, skipping zone signal")
            return

        # ── Step 3: R:R filter ─────────────────────────────────
        side = "long" if structure.trend == "bullish" else "short"
        balance = self.client.get_account_balance()

        setup = self.risk_manager.evaluate(
            coin=coin,
            side=side,
            current_price=current_price,
            zone=touched_zone,
            last_valid_high=structure.last_valid_high,
            last_valid_low=structure.last_valid_low,
            account_balance=balance,
        )

        if setup is None:
            logger.debug(f"{coin}: could not build trade setup")
            return

        await self._alert(
            f"📊 *{coin} Signal Detected*\n{setup.summary()}"
        )

        if not setup.approved:
            logger.info(f"{coin}: R:R {setup.rr_ratio} < {self.config.MIN_RR} — trade rejected")
            return

        # ── Execute trade ──────────────────────────────────────
        logger.info(f"Executing {side.upper()} trade on {coin} | R:R={setup.rr_ratio}")
        success = self.client.enter_trade(
            coin=coin,
            side=side,
            size_coin=setup.position_size_coin,
            stop_loss=setup.stop_loss,
            take_profit=setup.take_profit,
        )

        if success:
            state.in_trade = True
            state.trades_taken += 1
            await self._alert(
                f"✅ *Trade Entered — {coin}*\n{setup.summary()}"
            )
        else:
            await self._alert(f"❌ *Trade execution failed for {coin}*")

    # ──────────────────────────────────────────────────────────
    # Status helpers (used by Telegram bot)
    # ──────────────────────────────────────────────────────────

    def get_status_text(self) -> str:
        lines = ["*Strategy Status*\n"]
        for coin, state in self.pair_states.items():
            trend = state.structure.trend if state.structure else "unknown"
            zones = len(state.active_zones)
            lines.append(
                f"*{coin}*: trend=`{trend}` | zones={zones} | "
                f"trades={state.trades_taken}"
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
        """Sleep until the next candle closes (aligned to timeframe)."""
        now = time.time()
        remainder = now % self._interval_seconds
        return self._interval_seconds - remainder + 2  # +2s buffer

    @staticmethod
    def _timeframe_to_seconds(tf: str) -> int:
        mapping = {
            "1m": 60, "3m": 180, "5m": 300, "15m": 900,
            "30m": 1800, "1h": 3600, "4h": 14400, "1d": 86400,
        }
        return mapping.get(tf, 900)
