import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    # ── Telegram ───────────────────────────────────────────────
    TELEGRAM_BOT_TOKEN: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
    TELEGRAM_CHAT_ID: str = os.getenv("TELEGRAM_CHAT_ID", "")

    # ── Hyperliquid ────────────────────────────────────────────
    HL_PRIVATE_KEY: str = os.getenv("HL_PRIVATE_KEY", "")
    HL_WALLET_ADDRESS: str = os.getenv("HL_WALLET_ADDRESS", "")
    HL_TESTNET: bool = os.getenv("HL_TESTNET", "true").lower() == "true"

    # ── Strategy ───────────────────────────────────────────────
    PAIRS: list[str] = [p.strip() for p in os.getenv("PAIRS", "BTC,ETH,SOL").split(",")]
    TIMEFRAME: str = os.getenv("TIMEFRAME", "5m")
    POSITION_SIZE_PCT: float = float(os.getenv("POSITION_SIZE_PCT", "0.10"))
    MIN_RR: float = float(os.getenv("MIN_RR", "2.0"))
    PIVOT_LOOKBACK: int = int(os.getenv("PIVOT_LOOKBACK", "10"))

    # ── SMA Channel ────────────────────────────────────────────
    # SMA_LENGTH controls both SMA(High) and SMA(Low) for the price channel
    SMA_LENGTH: int = int(os.getenv("SMA_LENGTH", "20"))
    SL_BUFFER: float = float(os.getenv("SL_BUFFER", "0.001"))

    # ── Dry run / paper trading ────────────────────────────────
    # DRY_RUN=true  → live mainnet data, no wallet needed, trades simulated
    DRY_RUN: bool = os.getenv("DRY_RUN", "false").lower() == "true"
    PAPER_BALANCE: float = float(os.getenv("PAPER_BALANCE", "10000.0"))

    # ── Candle history ─────────────────────────────────────────
    # How many candles to load for structure analysis
    CANDLE_LIMIT: int = 200

    def validate(self) -> list[str]:
        """Return list of missing/invalid config values."""
        errors = []
        if not self.TELEGRAM_BOT_TOKEN:
            errors.append("TELEGRAM_BOT_TOKEN is not set")
        if not self.TELEGRAM_CHAT_ID:
            errors.append("TELEGRAM_CHAT_ID is not set")
        # Wallet credentials only required when not in dry run mode
        if not self.DRY_RUN:
            if not self.HL_PRIVATE_KEY:
                errors.append("HL_PRIVATE_KEY is not set (set DRY_RUN=true to skip)")
            if not self.HL_WALLET_ADDRESS:
                errors.append("HL_WALLET_ADDRESS is not set (set DRY_RUN=true to skip)")
        if self.TIMEFRAME not in ("1m", "3m", "5m", "15m", "30m", "1h", "4h"):
            errors.append(f"TIMEFRAME '{self.TIMEFRAME}' is not valid")
        return errors
