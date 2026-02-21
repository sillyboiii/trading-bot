"""
Hyperliquid Exchange Client
============================
Wraps the Hyperliquid Python SDK to provide clean, bot-friendly methods
for market data retrieval and order management.

Modes:
  DRY RUN  — live mainnet market data, no wallet needed, trades are simulated
  TESTNET  — real orders on testnet (https://api.hyperliquid-testnet.xyz)
  MAINNET  — real orders on mainnet (https://api.hyperliquid.xyz)
"""

import logging
import time
from typing import Optional

import pandas as pd
from hyperliquid.info import Info
from hyperliquid.utils import constants

logger = logging.getLogger(__name__)

# Slippage applied to simulated market orders (0.5%)
MARKET_SLIPPAGE = 0.005


class HyperliquidClient:
    def __init__(
        self,
        private_key: str = "",
        wallet_address: str = "",
        testnet: bool = True,
        dry_run: bool = False,
        paper_balance: float = 10_000.0,
    ):
        self.dry_run = dry_run
        self.testnet = testnet
        self.wallet_address = wallet_address.lower() if wallet_address else ""

        if dry_run:
            # Use mainnet for live read-only market data — no wallet needed
            self._info = Info(constants.MAINNET_API_URL, skip_ws=True)
            self._exchange = None
            self._paper_balance = paper_balance
            self._paper_positions: dict[str, dict] = {}
            logger.info(
                f"HyperliquidClient in DRY RUN mode "
                f"(paper balance: ${paper_balance:,.2f}, live mainnet data)"
            )
        else:
            from eth_account import Account
            from hyperliquid.exchange import Exchange

            base_url = constants.TESTNET_API_URL if testnet else constants.MAINNET_API_URL
            self._account = Account.from_key(private_key)
            self._info = Info(base_url, skip_ws=True)
            self._exchange = Exchange(self._account, base_url)
            net_label = "TESTNET" if testnet else "MAINNET"
            logger.info(
                f"HyperliquidClient initialised on {net_label} "
                f"for {wallet_address[:10]}…"
            )

    # ──────────────────────────────────────────────────────────
    # Market Data  (same in all modes — always real data)
    # ──────────────────────────────────────────────────────────

    def get_candles(
        self,
        coin: str,
        interval: str,
        limit: int = 200,
    ) -> pd.DataFrame:
        """
        Fetch recent OHLCV candles.

        Args:
            coin:     e.g. 'BTC', 'ETH', 'SOL'
            interval: e.g. '5m', '15m', '1h'
            limit:    number of candles to return

        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
            Sorted oldest → newest.
        """
        interval_ms = self._interval_to_ms(interval)
        end_ms = int(time.time() * 1000)
        start_ms = end_ms - (interval_ms * (limit + 5))

        raw = self._info.candles_snapshot(coin, interval, start_ms, end_ms)

        if not raw:
            logger.warning(f"No candle data returned for {coin} {interval}")
            return pd.DataFrame()

        records = [
            {
                "timestamp": int(c["t"]),
                "open":      float(c["o"]),
                "high":      float(c["h"]),
                "low":       float(c["l"]),
                "close":     float(c["c"]),
                "volume":    float(c["v"]),
            }
            for c in raw
        ]

        df = pd.DataFrame(records).sort_values("timestamp").reset_index(drop=True)
        return df.tail(limit).reset_index(drop=True)

    def get_current_price(self, coin: str) -> Optional[float]:
        """Return the current mid price for a coin."""
        try:
            mids = self._info.all_mids()
            price_str = mids.get(coin)
            return float(price_str) if price_str else None
        except Exception as e:
            logger.error(f"Error fetching price for {coin}: {e}")
            return None

    # ──────────────────────────────────────────────────────────
    # Account  (paper simulation in dry run)
    # ──────────────────────────────────────────────────────────

    def get_account_balance(self) -> float:
        """Return current account value in USD."""
        if self.dry_run:
            return self._paper_balance

        try:
            state = self._info.user_state(self.wallet_address)
            return float(state["marginSummary"]["accountValue"])
        except Exception as e:
            logger.error(f"Error fetching account balance: {e}")
            return 0.0

    def get_positions(self) -> list[dict]:
        """Return all open perpetual positions."""
        if self.dry_run:
            # Compute live unrealized PnL against current mid price
            result = []
            for pos in self._paper_positions.values():
                price = self.get_current_price(pos["coin"]) or pos["entry_price"]
                if pos["side"] == "long":
                    upnl = (price - pos["entry_price"]) * pos["size"]
                else:
                    upnl = (pos["entry_price"] - price) * pos["size"]
                result.append({**pos, "unrealized_pnl": upnl, "current_price": price})
            return result

        try:
            state = self._info.user_state(self.wallet_address)
            positions = []
            for pos in state.get("assetPositions", []):
                p = pos.get("position", {})
                size = float(p.get("szi", 0))
                if size != 0:
                    positions.append(
                        {
                            "coin":           p["coin"],
                            "side":           "long" if size > 0 else "short",
                            "size":           abs(size),
                            "entry_price":    float(p.get("entryPx", 0)),
                            "unrealized_pnl": float(p.get("unrealizedPnl", 0)),
                            "liquidation_px": float(p.get("liquidationPx") or 0),
                        }
                    )
            return positions
        except Exception as e:
            logger.error(f"Error fetching positions: {e}")
            return []

    def get_open_orders(self, coin: Optional[str] = None) -> list[dict]:
        """Return open orders, optionally filtered by coin."""
        if self.dry_run:
            return []  # Paper trades have no exchange orders

        try:
            orders = self._info.open_orders(self.wallet_address)
            if coin:
                orders = [o for o in orders if o.get("coin") == coin]
            return orders
        except Exception as e:
            logger.error(f"Error fetching open orders: {e}")
            return []

    def has_open_position(self, coin: str) -> bool:
        if self.dry_run:
            return coin in self._paper_positions
        return any(p["coin"] == coin for p in self.get_positions())

    # ──────────────────────────────────────────────────────────
    # Order Execution  (paper simulation in dry run)
    # ──────────────────────────────────────────────────────────

    def enter_trade(
        self,
        coin: str,
        side: str,
        size_coin: float,
        stop_loss: float,
        take_profit: float,
    ) -> bool:
        """
        Enter a trade with automatic SL and TP orders.

        In dry run mode this records a paper position; no real orders are sent.

        Args:
            coin:        e.g. 'BTC'
            side:        'long' or 'short'
            size_coin:   position size in base currency
            stop_loss:   stop loss price
            take_profit: take profit price

        Returns:
            True if the trade was entered (or simulated) successfully.
        """
        current_price = self.get_current_price(coin)
        if not current_price:
            logger.error(f"Cannot enter {coin} trade: price unavailable")
            return False

        if self.dry_run:
            self._paper_positions[coin] = {
                "coin":           coin,
                "side":           side,
                "size":           size_coin,
                "entry_price":    current_price,
                "stop_loss":      stop_loss,
                "take_profit":    take_profit,
                "unrealized_pnl": 0.0,
                "liquidation_px": 0.0,
            }
            logger.info(
                f"[DRY RUN] Paper {side.upper()} on {coin} | "
                f"entry=${current_price:,.4f} | sz={size_coin:.6f} | "
                f"SL=${stop_loss:,.4f} | TP=${take_profit:,.4f}"
            )
            return True

        # ── Real execution ──────────────────────────────────────
        is_buy = side == "long"
        if is_buy:
            limit_px = round(current_price * (1 + MARKET_SLIPPAGE), 6)
        else:
            limit_px = round(current_price * (1 - MARKET_SLIPPAGE), 6)

        logger.info(
            f"Placing {side.upper()} entry on {coin} | "
            f"size={size_coin:.6f} | price≈${limit_px:,.4f}"
        )

        entry_result = self._exchange.order(
            coin=coin,
            is_buy=is_buy,
            sz=round(size_coin, 6),
            limit_px=limit_px,
            order_type={"limit": {"tif": "Ioc"}},
            reduce_only=False,
        )

        if not self._order_ok(entry_result):
            logger.error(f"Entry order failed for {coin}: {entry_result}")
            return False

        logger.info(f"Entry order filled for {coin}")

        sl_placed = self._place_trigger_order(
            coin=coin,
            is_buy=not is_buy,
            sz=size_coin,
            trigger_px=stop_loss,
            tpsl_type="sl",
        )
        tp_placed = self._place_limit_order(
            coin=coin,
            is_buy=not is_buy,
            sz=size_coin,
            limit_px=take_profit,
            reduce_only=True,
        )

        if not sl_placed:
            logger.warning(f"SL order failed for {coin} — monitor manually!")
        if not tp_placed:
            logger.warning(f"TP order failed for {coin} — monitor manually!")

        return True

    def close_position(self, coin: str) -> bool:
        """Market-close an open position."""
        if self.dry_run:
            pos = self._paper_positions.pop(coin, None)
            if not pos:
                logger.warning(f"[DRY RUN] No paper position for {coin}")
                return False
            current_price = self.get_current_price(coin) or pos["entry_price"]
            if pos["side"] == "long":
                pnl = (current_price - pos["entry_price"]) * pos["size"]
            else:
                pnl = (pos["entry_price"] - current_price) * pos["size"]
            self._paper_balance += pnl
            logger.info(
                f"[DRY RUN] Closed {coin} | PnL=${pnl:+,.2f} | "
                f"balance=${self._paper_balance:,.2f}"
            )
            return True

        positions = self.get_positions()
        pos = next((p for p in positions if p["coin"] == coin), None)
        if not pos:
            logger.warning(f"No open position for {coin}")
            return False

        is_buy = pos["side"] == "short"
        current_price = self.get_current_price(coin)
        if not current_price:
            return False

        if is_buy:
            limit_px = round(current_price * (1 + MARKET_SLIPPAGE), 6)
        else:
            limit_px = round(current_price * (1 - MARKET_SLIPPAGE), 6)

        result = self._exchange.order(
            coin=coin,
            is_buy=is_buy,
            sz=round(pos["size"], 6),
            limit_px=limit_px,
            order_type={"limit": {"tif": "Ioc"}},
            reduce_only=True,
        )

        ok = self._order_ok(result)
        if ok:
            logger.info(f"Closed {coin} position")
            self.cancel_all_orders(coin)
        return ok

    def cancel_all_orders(self, coin: str) -> bool:
        """Cancel all open orders for a coin."""
        if self.dry_run:
            return True  # No real orders to cancel

        orders = self.get_open_orders(coin)
        if not orders:
            return True

        oids = [{"coin": o["coin"], "oid": o["oid"]} for o in orders]
        try:
            result = self._exchange.bulk_cancel(oids)
            return self._order_ok(result)
        except Exception as e:
            logger.error(f"Error cancelling orders for {coin}: {e}")
            return False

    # ──────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────

    def _place_trigger_order(
        self,
        coin: str,
        is_buy: bool,
        sz: float,
        trigger_px: float,
        tpsl_type: str,
    ) -> bool:
        try:
            result = self._exchange.order(
                coin=coin,
                is_buy=is_buy,
                sz=round(sz, 6),
                limit_px=trigger_px,
                order_type={
                    "trigger": {
                        "triggerPx": trigger_px,
                        "isMarket": True,
                        "tpsl": tpsl_type,
                    }
                },
                reduce_only=True,
            )
            return self._order_ok(result)
        except Exception as e:
            logger.error(f"Error placing trigger order: {e}")
            return False

    def _place_limit_order(
        self,
        coin: str,
        is_buy: bool,
        sz: float,
        limit_px: float,
        reduce_only: bool = False,
    ) -> bool:
        try:
            result = self._exchange.order(
                coin=coin,
                is_buy=is_buy,
                sz=round(sz, 6),
                limit_px=limit_px,
                order_type={"limit": {"tif": "Gtc"}},
                reduce_only=reduce_only,
            )
            return self._order_ok(result)
        except Exception as e:
            logger.error(f"Error placing limit order: {e}")
            return False

    @staticmethod
    def _order_ok(result: dict) -> bool:
        if not result:
            return False
        if result.get("status") == "ok":
            return True
        response = result.get("response", {})
        if isinstance(response, dict):
            data = response.get("data", {})
            statuses = data.get("statuses", [])
            for s in statuses:
                if "error" in s:
                    logger.error(f"Order error: {s['error']}")
                    return False
            return bool(statuses)
        return False

    @staticmethod
    def _interval_to_ms(interval: str) -> int:
        mapping = {
            "1m":  60_000,
            "3m":  180_000,
            "5m":  300_000,
            "15m": 900_000,
            "30m": 1_800_000,
            "1h":  3_600_000,
            "4h":  14_400_000,
            "1d":  86_400_000,
        }
        return mapping.get(interval, 900_000)
