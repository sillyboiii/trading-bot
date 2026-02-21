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
    POSITION_SIZE_PCT: float = float(os.getenv("POSITION_SIZE_PCT", "0.10"))  # legacy fallback
    RISK_PER_TRADE_PCT: float = float(os.getenv("RISK_PER_TRADE_PCT", "0.01"))  # 1% balance at risk per trade
    MAX_POSITION_PCT: float = float(os.getenv("MAX_POSITION_PCT", "0.25"))    # cap at 25% of balance
    MIN_RR: float = float(os.getenv("MIN_RR", "2.5"))
    PIVOT_LOOKBACK: int = int(os.getenv("PIVOT_LOOKBACK", "10"))

    # ── SMA Channel ────────────────────────────────────────────
    SMA_LENGTH: int = int(os.getenv("SMA_LENGTH", "20"))
    SL_BUFFER: float = float(os.getenv("SL_BUFFER", "0.001"))

    # ── ATR-based stop loss ────────────────────────────────────
    # SL is placed ATR_SL_MULT × ATR(ATR_LENGTH) from entry.
    # Adapts to actual volatility — stops won't get hit by normal candle noise.
    ATR_LENGTH: int = int(os.getenv("ATR_LENGTH", "14"))
    ATR_SL_MULT: float = float(os.getenv("ATR_SL_MULT", "1.5"))

    # ── Signal filters ─────────────────────────────────────────
    TREND_CONFIRM_CANDLES: int = int(os.getenv("TREND_CONFIRM_CANDLES", "5"))
    PULLBACK_ONLY: bool = os.getenv("PULLBACK_ONLY", "true").lower() == "true"
    # Volume gate — 0 = disabled, 1.2 = must be 20% above avg. Disabled by
    # default: added no edge but cut valid setups on 5m charts.
    VOLUME_MULT: float = float(os.getenv("VOLUME_MULT", "0"))
    # Macro EMA — 0 = disabled. Disabled by default: caused inverted filtering
    # on pullback entries (price is naturally below the EMA during the dip).
    MACRO_EMA: int = int(os.getenv("MACRO_EMA", "0"))

    # ── Trade management ───────────────────────────────────────
    # Move SL to entry once price reaches this many R in profit (0 = disabled).
    # Set to 1.5 — at 1.0 the breakeven fired too early, converting wins into
    # breakevens before the trade had room to reach TP.
    BREAKEVEN_AT_R: float = float(os.getenv("BREAKEVEN_AT_R", "1.5"))

    # ── Circuit breaker ────────────────────────────────────────
    # Pause trading for CIRCUIT_PAUSE_HOURS after either:
    #   - CONSECUTIVE_LOSS_LIMIT consecutive losses, OR
    #   - MAX_DAILY_LOSS_PCT drawdown from the session-high balance
    CONSECUTIVE_LOSS_LIMIT: int = int(os.getenv("CONSECUTIVE_LOSS_LIMIT", "3"))
    MAX_DAILY_LOSS_PCT: float = float(os.getenv("MAX_DAILY_LOSS_PCT", "0.03"))  # 3%
    CIRCUIT_PAUSE_HOURS: float = float(os.getenv("CIRCUIT_PAUSE_HOURS", "6.0"))

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
