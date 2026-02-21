"""
Telegram Bot Interface
=======================
Commands available to the trader:

  /start    — Start the trading bot
  /stop     — Stop the trading bot (open positions stay open)
  /status   — Show trend + zones for all pairs
  /balance  — Account balance
  /positions — Open positions
  /close <coin> — Manually close a position
  /orders   — Open orders
  /config   — Show current config
  /backtest — Run a quick backtest summary
  /help     — Command list
"""

import logging
from typing import Optional

from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
)
from telegram.constants import ParseMode

from config import Config
from src.exchange.hyperliquid_client import HyperliquidClient
from src.strategy.engine import StrategyEngine
from src.backtester.backtest import Backtester

logger = logging.getLogger(__name__)


class TradingBot:
    """
    Wraps python-telegram-bot and owns the StrategyEngine lifecycle.
    """

    def __init__(self, config: Config):
        self.config = config
        self.client: Optional[HyperliquidClient] = None
        self.engine: Optional[StrategyEngine] = None
        self._app: Optional[Application] = None
        self._engine_task = None

    # ──────────────────────────────────────────────────────────
    # Entry point
    # ──────────────────────────────────────────────────────────

    def run(self):
        """Build and run the Telegram application (blocking)."""
        errors = self.config.validate()
        if errors:
            for e in errors:
                logger.error(f"Config error: {e}")
            raise SystemExit("Fix configuration errors before starting.")

        self.client = HyperliquidClient(
            private_key=self.config.HL_PRIVATE_KEY,
            wallet_address=self.config.HL_WALLET_ADDRESS,
            testnet=self.config.HL_TESTNET,
            dry_run=self.config.DRY_RUN,
            paper_balance=self.config.PAPER_BALANCE,
        )

        self._app = (
            Application.builder()
            .token(self.config.TELEGRAM_BOT_TOKEN)
            .build()
        )

        # Register command handlers
        handlers = [
            ("start",     self._cmd_start),
            ("stop",      self._cmd_stop),
            ("status",    self._cmd_status),
            ("balance",   self._cmd_balance),
            ("positions", self._cmd_positions),
            ("close",     self._cmd_close),
            ("orders",    self._cmd_orders),
            ("config",    self._cmd_config),
            ("backtest",  self._cmd_backtest),
            ("help",      self._cmd_help),
        ]
        for name, handler in handlers:
            self._app.add_handler(CommandHandler(name, handler))

        logger.info("Telegram bot starting…")
        self._app.run_polling(drop_pending_updates=True)

    # ──────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────

    def _is_authorised(self, update: Update) -> bool:
        """Only respond to the configured chat."""
        return str(update.effective_chat.id) == str(self.config.TELEGRAM_CHAT_ID)

    async def _reply(self, update: Update, text: str):
        await update.message.reply_text(text, parse_mode=ParseMode.MARKDOWN)

    async def _send_alert(self, msg: str):
        """Called by the strategy engine to push notifications."""
        if self._app:
            await self._app.bot.send_message(
                chat_id=self.config.TELEGRAM_CHAT_ID,
                text=msg,
                parse_mode=ParseMode.MARKDOWN,
            )

    def _build_engine(self) -> StrategyEngine:
        return StrategyEngine(
            config=self.config,
            client=self.client,
            alert_callback=self._send_alert,
        )

    # ──────────────────────────────────────────────────────────
    # Command handlers
    # ──────────────────────────────────────────────────────────

    async def _cmd_start(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        if not self._is_authorised(update):
            return

        if self.engine and self.engine.running:
            await self._reply(update, "⚠️ Bot is already running.")
            return

        self.engine = self._build_engine()
        self._engine_task = ctx.application.create_task(self.engine.start())
        await self._reply(update, "✅ *Trading bot started!* I'll alert you on every signal.")

    async def _cmd_stop(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        if not self._is_authorised(update):
            return

        if not self.engine or not self.engine.running:
            await self._reply(update, "ℹ️ Bot is not running.")
            return

        await self.engine.stop()
        if self._engine_task:
            self._engine_task.cancel()
            self._engine_task = None
        await self._reply(update, "⛔ *Bot stopped.* Existing positions are untouched.")

    async def _cmd_status(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        if not self._is_authorised(update):
            return

        if not self.engine:
            await self._reply(update, "ℹ️ Bot is not running. Use /start first.")
            return

        await self._reply(update, self.engine.get_status_text())

    async def _cmd_balance(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        if not self._is_authorised(update):
            return

        balance = self.client.get_account_balance()
        if self.config.DRY_RUN:
            label = "DRY RUN (paper)"
        elif self.config.HL_TESTNET:
            label = "TESTNET"
        else:
            label = "MAINNET"
        await self._reply(update, f"💰 *Balance ({label})*\n`${balance:,.2f} USD`")

    async def _cmd_positions(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        if not self._is_authorised(update):
            return

        positions = self.client.get_positions()
        if not positions:
            await self._reply(update, "📭 No open positions.")
            return

        lines = ["*Open Positions*\n"]
        for p in positions:
            pnl_sign = "+" if p["unrealized_pnl"] >= 0 else ""
            lines.append(
                f"*{p['coin']}* ({p['side'].upper()})\n"
                f"  Size:  {p['size']:.6f}\n"
                f"  Entry: ${p['entry_price']:,.4f}\n"
                f"  PnL:   {pnl_sign}${p['unrealized_pnl']:,.2f}\n"
            )
        await self._reply(update, "\n".join(lines))

    async def _cmd_close(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        if not self._is_authorised(update):
            return

        args = ctx.args
        if not args:
            await self._reply(update, "Usage: /close BTC")
            return

        coin = args[0].upper()
        await self._reply(update, f"⏳ Closing {coin} position…")
        success = self.client.close_position(coin)

        if success:
            if self.engine and coin in self.engine.pair_states:
                self.engine.pair_states[coin].in_trade = False
            await self._reply(update, f"✅ *{coin}* position closed.")
        else:
            await self._reply(update, f"❌ Failed to close {coin} — check logs.")

    async def _cmd_orders(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        if not self._is_authorised(update):
            return

        orders = self.client.get_open_orders()
        if not orders:
            await self._reply(update, "📭 No open orders.")
            return

        lines = ["*Open Orders*\n"]
        for o in orders:
            lines.append(
                f"{o.get('coin')} | {o.get('side', '?').upper()} | "
                f"${float(o.get('limitPx', 0)):,.4f} | "
                f"sz={o.get('sz', '?')}"
            )
        await self._reply(update, "\n".join(lines))

    async def _cmd_config(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        if not self._is_authorised(update):
            return

        cfg = self.config
        if cfg.DRY_RUN:
            net = f"DRY RUN 🧪 (paper ${cfg.PAPER_BALANCE:,.0f})"
        elif cfg.HL_TESTNET:
            net = "TESTNET ⚠️"
        else:
            net = "MAINNET"
        text = (
            f"*Current Config*\n"
            f"Mode:        `{net}`\n"
            f"Pairs:       `{', '.join(cfg.PAIRS)}`\n"
            f"Timeframe:   `{cfg.TIMEFRAME}`\n"
            f"SMA length:  `{cfg.SMA_LENGTH}`\n"
            f"Position sz: `{cfg.POSITION_SIZE_PCT * 100:.0f}%`\n"
            f"Min R:R:     `{cfg.MIN_RR}`\n"
            f"Pivot lb:    `{cfg.PIVOT_LOOKBACK}`\n"
            f"SL buffer:   `{cfg.SL_BUFFER * 100:.1f}%`\n"
        )
        await self._reply(update, text)

    async def _cmd_backtest(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        if not self._is_authorised(update):
            return

        await self._reply(update, "⏳ Running backtest… this may take a moment.")

        backtester = Backtester(
            client=self.client,
            config=self.config,
        )

        results = backtester.run()
        if not results:
            await self._reply(update, "❌ Backtest failed — check logs.")
            return

        lines = ["*Backtest Results*\n"]
        for coin, tf_results in results.items():
            lines.append(f"*{coin}*")
            for tf, res in tf_results.items():
                lines.append(
                    f"  `{tf}` → trades={res['total_trades']} | "
                    f"win={res['win_rate']:.1f}% | "
                    f"avg R:R={res['avg_rr']:.2f} | "
                    f"rejected={res['rejected']}"
                )
            lines.append("")
        await self._reply(update, "\n".join(lines))

    async def _cmd_help(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        if not self._is_authorised(update):
            return

        text = (
            "*Available Commands*\n\n"
            "/start — Start trading\n"
            "/stop — Stop trading\n"
            "/status — Strategy status per pair\n"
            "/balance — Account balance\n"
            "/positions — Open positions\n"
            "/close `<COIN>` — Close a position manually\n"
            "/orders — Open orders\n"
            "/config — Show current settings\n"
            "/backtest — Backtest the strategy\n"
            "/help — This message\n"
        )
        await self._reply(update, text)
