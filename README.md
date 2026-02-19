# Price Action Trading Bot
### Hyperliquid Perpetuals × Telegram Control

A fully automated trading bot implementing a 3-step price action strategy:

1. **Market Structure** — valid highs/lows, trend detection (BOS-based)
2. **Supply & Demand Zones** — impulse-based zone detection, trend-aligned only
3. **R:R Filter** — only trades with ≥ 2.5:1 risk-to-reward are executed

---

## Project Structure

```
trading-bot/
├── main.py                        ← Entry point
├── config.py                      ← Config loader
├── requirements.txt
├── .env.example                   ← Copy this → .env
└── src/
    ├── strategy/
    │   ├── market_structure.py    ← Step 1: Trend detection
    │   ├── zones.py               ← Step 2: Supply/demand zones
    │   ├── risk_manager.py        ← Step 3: SL/TP/R:R
    │   └── engine.py              ← Strategy orchestrator
    ├── exchange/
    │   └── hyperliquid_client.py  ← Hyperliquid API wrapper
    ├── bot/
    │   └── telegram_bot.py        ← Telegram command interface
    └── backtester/
        └── backtest.py            ← Historical strategy simulation
```

---

## Quick Start (Local)

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set up environment

```bash
cp .env.example .env
```

Edit `.env` and fill in:

| Variable | Where to get it |
|---|---|
| `TELEGRAM_BOT_TOKEN` | Message [@BotFather](https://t.me/BotFather) on Telegram, create a new bot |
| `TELEGRAM_CHAT_ID` | Message [@userinfobot](https://t.me/userinfobot) on Telegram |
| `HL_PRIVATE_KEY` | Export from your Hyperliquid-connected wallet (MetaMask etc.) |
| `HL_WALLET_ADDRESS` | Your public wallet address |
| `HL_TESTNET` | Set `true` to start on testnet (recommended) |

### 3. Fund testnet account

1. Go to [app.hyperliquid-testnet.xyz](https://app.hyperliquid-testnet.xyz)
2. Connect your wallet
3. Use the testnet faucet to get test USDC

### 4. Run

```bash
python main.py
```

Open Telegram and send your bot `/start`.

---

## Telegram Commands

| Command | Description |
|---|---|
| `/start` | Start the trading bot |
| `/stop` | Stop trading (positions stay open) |
| `/status` | Trend + zones per pair |
| `/balance` | Account balance |
| `/positions` | Open positions + unrealised PnL |
| `/close BTC` | Manually close a position |
| `/orders` | Open orders (SL/TP) |
| `/config` | Show current settings |
| `/backtest` | Run historical simulation (5m vs 15m comparison) |
| `/help` | Command list |

---

## Free 24/7 Hosting — Oracle Cloud (Always Free)

Oracle Cloud gives you **2 free VMs that never expire** — ideal for a trading bot.

1. Sign up at [cloud.oracle.com](https://cloud.oracle.com) (requires credit card for verification, but you won't be charged on Always Free)
2. Create a VM:
   - Shape: `VM.Standard.E2.1.Micro` (1 OCPU, 1GB RAM — Always Free)
   - OS: Ubuntu 22.04
3. SSH into your VM and run:

```bash
# Install Python
sudo apt update && sudo apt install python3-pip python3-venv git -y

# Clone your repo
git clone <your-repo-url>
cd trading-bot

# Set up environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
nano .env   # fill in your values

# Run in background (survives SSH disconnect)
nohup python main.py > logs/bot.log 2>&1 &

# Or use screen for easy reattachment:
screen -S tradingbot
python main.py
# Ctrl+A, D to detach. screen -r tradingbot to reattach.
```

---

## Strategy Configuration

Edit `.env` to tune the strategy:

```env
TIMEFRAME=15m          # 5m is more signals, 15m is higher quality
MIN_RR=2.5             # Minimum risk-to-reward (strategy rule)
POSITION_SIZE_PCT=0.10 # 10% of account per trade
PIVOT_LOOKBACK=5       # Candles on each side to confirm a swing
SL_BUFFER=0.001        # 0.1% extra buffer below/above zone for SL
```

---

## Going Live (Mainnet)

Once you are satisfied with testnet results:

1. Set `HL_TESTNET=false` in `.env`
2. Fund your mainnet Hyperliquid account with real USDC
3. Start small — reduce `POSITION_SIZE_PCT` (e.g. `0.02`) initially
4. Monitor via `/positions` and `/balance` in Telegram

---

## Risk Warning

This bot trades real money on your behalf. Always:
- Start on **testnet**
- Use a dedicated wallet (not your main wallet)
- Never store more than you can afford to lose
- Monitor open positions regularly
