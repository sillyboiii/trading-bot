"""
Trading Bot — Entry Point
=========================
Usage:
    python main.py

Ensure you have copied .env.example → .env and filled in your values.
"""

import logging
import sys
from pathlib import Path

from config import Config
from src.bot.telegram_bot import TradingBot

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


def main():
    logger.info("=" * 60)
    logger.info("  Price Action Trading Bot  ")
    logger.info("  Hyperliquid + Telegram    ")
    logger.info("=" * 60)

    config = Config()

    errors = config.validate()
    if errors:
        logger.error("Configuration errors found:")
        for e in errors:
            logger.error(f"  - {e}")
        logger.error("Please copy .env.example → .env and fill in your values.")
        sys.exit(1)

    net = "TESTNET ⚠️" if config.HL_TESTNET else "MAINNET"
    logger.info(f"Network:    {net}")
    logger.info(f"Pairs:      {config.PAIRS}")
    logger.info(f"Timeframe:  {config.TIMEFRAME}")
    logger.info(f"Min R:R:    {config.MIN_RR}")
    logger.info(f"Position:   {config.POSITION_SIZE_PCT * 100:.0f}% of account")

    bot = TradingBot(config)
    bot.run()


if __name__ == "__main__":
    main()
