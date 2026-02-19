"""
Backtester
==========
Runs the full 3-step strategy on historical candles for each pair
and reports key performance metrics:

  - Total trades taken
  - Trades rejected (R:R < MIN_RR)
  - Win rate  (trade reached TP before SL)
  - Average realised R:R
  - Net return (% of account)

No actual orders are placed — simulation only.
"""

import logging
from typing import Optional

import pandas as pd

from config import Config
from src.exchange.hyperliquid_client import HyperliquidClient
from src.strategy.market_structure import MarketStructure
from src.strategy.zones import ZoneDetector
from src.strategy.risk_manager import RiskManager

logger = logging.getLogger(__name__)

# Backtest uses this many candles of history
BACKTEST_CANDLE_LIMIT = 500

# Simulated starting balance for metric calculations
SIM_BALANCE = 10_000.0


class Backtester:
    def __init__(self, client: HyperliquidClient, config: Config):
        self.client = client
        self.config = config

        self.ms = MarketStructure(pivot_lookback=config.PIVOT_LOOKBACK)
        self.zones = ZoneDetector(
            impulse_body_ratio=config.IMPULSE_BODY_RATIO,
            impulse_size_multiplier=config.IMPULSE_SIZE_MULTIPLIER,
        )
        self.risk = RiskManager(
            min_rr=config.MIN_RR,
            position_size_pct=config.POSITION_SIZE_PCT,
            sl_buffer=config.SL_BUFFER,
        )

    # ──────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────

    def run(self) -> dict:
        """
        Run backtest for all configured pairs and timeframes.
        Returns a dict keyed by coin with result dicts.
        """
        results = {}
        timeframes = [self.config.TIMEFRAME]

        # Also compare 5m vs 15m if user is on one of them
        if self.config.TIMEFRAME == "15m":
            timeframes = ["5m", "15m"]
        elif self.config.TIMEFRAME == "5m":
            timeframes = ["5m", "15m"]

        for coin in self.config.PAIRS:
            results[coin] = {}
            for tf in timeframes:
                logger.info(f"Backtesting {coin} on {tf}…")
                df = self.client.get_candles(coin, tf, BACKTEST_CANDLE_LIMIT)
                if df.empty:
                    logger.warning(f"No data for {coin} {tf}")
                    continue
                res = self._backtest_pair(coin, df)
                results[coin][tf] = res
                logger.info(
                    f"{coin} {tf}: trades={res['total_trades']} "
                    f"winrate={res['win_rate']:.1f}% "
                    f"avg_rr={res['avg_rr']:.2f}"
                )

        return results

    # ──────────────────────────────────────────────────────────
    # Core simulation
    # ──────────────────────────────────────────────────────────

    def _backtest_pair(self, coin: str, df: pd.DataFrame) -> dict:
        """
        Walk-forward simulation:
        At each candle, re-run the strategy on all data up to that
        point (excluding the current candle) and check for entry.
        Then simulate the trade outcome on subsequent candles.
        """
        total_trades = 0
        rejected = 0
        wins = 0
        losses = 0
        rr_values: list[float] = []
        net_pct = 0.0
        balance = SIM_BALANCE

        # Start from enough history for the strategy to have data
        min_start = self.config.PIVOT_LOOKBACK * 4 + 20
        # Track which candle the current simulated trade started at
        in_trade_until: Optional[int] = None

        for i in range(min_start, len(df) - 1):
            if in_trade_until is not None and i <= in_trade_until:
                continue  # Still in a simulated trade

            # Data available up to and including candle i-1 (closed)
            window = df.iloc[:i].copy().reset_index(drop=True)

            # Step 1: structure
            structure = self.ms.analyse(window)
            if structure.trend == "ranging":
                continue

            # Step 2: zones
            active_zones = self.zones.get_active_zones(window, structure.trend)
            if not active_zones:
                continue

            # Step 3: check if price at open of candle i touches a zone
            candle = df.iloc[i]
            entry_price = float(candle["open"])

            touched = self.zones.price_in_zone(entry_price, active_zones)
            if not touched:
                continue

            side = "long" if structure.trend == "bullish" else "short"
            setup = self.risk.evaluate(
                coin=coin,
                side=side,
                current_price=entry_price,
                zone=touched,
                last_valid_high=structure.last_valid_high,
                last_valid_low=structure.last_valid_low,
                account_balance=balance,
            )

            if setup is None:
                continue

            if not setup.approved:
                rejected += 1
                continue

            total_trades += 1

            # Simulate trade outcome on future candles
            outcome = self._simulate_outcome(df, i + 1, setup)

            if outcome == "win":
                wins += 1
                rr_values.append(setup.rr_ratio)
                pnl = setup.position_size_usd * setup.rr_ratio * self.config.POSITION_SIZE_PCT
                net_pct += self.config.POSITION_SIZE_PCT * setup.rr_ratio * 100
            elif outcome == "loss":
                losses += 1
                rr_values.append(-1.0)
                net_pct -= self.config.POSITION_SIZE_PCT * 100

            # Estimate how long trade lasted (crude: assume 10 candles max)
            in_trade_until = i + 20

        win_rate = (wins / total_trades * 100) if total_trades > 0 else 0.0
        avg_rr = (sum(rr_values) / len(rr_values)) if rr_values else 0.0

        return {
            "total_trades": total_trades,
            "wins": wins,
            "losses": losses,
            "rejected": rejected,
            "win_rate": round(win_rate, 1),
            "avg_rr": round(avg_rr, 2),
            "net_pct": round(net_pct, 2),
        }

    def _simulate_outcome(
        self,
        df: pd.DataFrame,
        start_idx: int,
        setup,
        max_candles: int = 50,
    ) -> str:
        """
        Walk forward from start_idx and return 'win', 'loss', or 'timeout'.
        A win is when price touches TP before SL.
        """
        for i in range(start_idx, min(start_idx + max_candles, len(df))):
            high = float(df["high"].iloc[i])
            low = float(df["low"].iloc[i])

            if setup.side == "long":
                if low <= setup.stop_loss:
                    return "loss"
                if high >= setup.take_profit:
                    return "win"
            else:
                if high >= setup.stop_loss:
                    return "loss"
                if low <= setup.take_profit:
                    return "win"

        return "timeout"
