"""
Trading Bot — Entry Point
=========================
Usage:
    python main.py              # run the Telegram bot
    python main.py --backtest   # run backtest and print results, then exit

Ensure you have copied .env.example → .env and filled in your values.
"""

import argparse
import logging
import sys
from pathlib import Path

from config import Config
from src.exchange.hyperliquid_client import HyperliquidClient

# ── Logging setup ──────────────────────────────────────────────────────────────
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_dir / "bot.log"),
    ],
)

# Silence noisy third-party loggers
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("telegram").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


def run_backtest(config: Config):
    """Run backtest and print results to stdout, then exit."""
    client = HyperliquidClient(
        private_key=config.HL_PRIVATE_KEY,
        wallet_address=config.HL_WALLET_ADDRESS,
        testnet=config.HL_TESTNET,
        dry_run=True,  # backtest never needs a real wallet
        paper_balance=config.PAPER_BALANCE,
    )

    from src.backtester.backtest import Backtester

    logger.info("Starting backtest — SMA Channel Strategy")
    logger.info(f"Pairs: {config.PAIRS} | SMA: {config.SMA_LENGTH} | Min R:R: {config.MIN_RR}")

    backtester = Backtester(client=client, config=config)
    results = backtester.run()

    print("\n" + "=" * 72)
    print("  BACKTEST RESULTS — SMA Channel Strategy")
    print("  avg_win_rr = R:R of winners only  |  EV = expected value per trade in R")
    print("=" * 72)
    for coin, tf_results in results.items():
        print(f"\n  {coin}")
        print(f"  {'─' * 64}")
        for tf, res in tf_results.items():
            profitable = "✓" if res['net_pct'] > 0 else "✗"
            print(
                f"  {profitable} {tf:>4s} │ "
                f"trades={res['total_trades']:>3d} ({res['wins']}W/{res['losses']}L/{res['timeouts']}T) │ "
                f"win={res['win_rate']:>5.1f}% │ "
                f"win R:R={res['avg_win_rr']:>5.2f} │ "
                f"EV={res['ev_per_trade']:>+6.3f}R │ "
                f"net={res['net_pct']:>+7.1f}%"
            )
    print("=" * 72 + "\n")


def run_bot(config: Config):
    """Start the Telegram trading bot (blocking)."""
    from src.bot.telegram_bot import TradingBot

    logger.info("=" * 60)
    logger.info("  SMA Channel Trading Bot    ")
    logger.info("  Hyperliquid + Telegram     ")
    logger.info("=" * 60)

    net = "TESTNET ⚠️" if config.HL_TESTNET else "MAINNET"
    logger.info(f"Network:    {net}")
    logger.info(f"Pairs:      {config.PAIRS}")
    logger.info(f"Timeframe:  {config.TIMEFRAME}")
    logger.info(f"SMA length: {config.SMA_LENGTH}")
    logger.info(f"Min R:R:    {config.MIN_RR}")
    logger.info(f"Position:   {config.POSITION_SIZE_PCT * 100:.0f}% of account")

    bot = TradingBot(config)
    bot.run()


def main():
    parser = argparse.ArgumentParser(description="SMA Channel Trading Bot")
    parser.add_argument(
        "--backtest",
        action="store_true",
        help="Run a backtest on historical data and exit",
    )
    args = parser.parse_args()

    config = Config()

    # For backtest mode, wallet keys are not required
    if args.backtest:
        # Only validate Telegram creds (not needed for backtest either, skip all)
        run_backtest(config)
        return

    errors = config.validate()
    if errors:
        logger.error("Configuration errors found:")
        for e in errors:
            logger.error(f"  - {e}")
        logger.error("Please copy .env.example → .env and fill in your values.")
        sys.exit(1)

    run_bot(config)


if __name__ == "__main__":
    main()
