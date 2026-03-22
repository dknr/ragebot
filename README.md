# RageBot

Sentiment Analysis Robot for Telegram


## Installation
Copy `.env.example` to `.env` and update with your token
```bash
cp .env.example .env
# Edit .env and add your TELEGRAM_BOT_TOKEN
```

## Running Manually

```bash
source venv/bin/activate
python main.py
```

## Installing as System Service

1. Install the systemd unit file:

```bash
# Replace with your actual paths
sed -i 's|{{RAGEBOT_DIR}}|/path/to/ragebot|g' ragebot.service
sed -i 's|{{VENV_PATH}}|/path/to/ragebot/venv|g' ragebot.service

sudo cp ragebot.service /etc/systemd/system/
sudo systemctl daemon-reload
```

2. Start and enable the service:

```bash
sudo systemctl start ragebot
sudo systemctl enable ragebot
```

3. Check service status:

```bash
sudo systemctl status ragebot
journalctl -u ragebot -f
```

## Bot Commands

- `/ragebot_start` - Show welcome message
- `/ragebot_help` - Show help
- `/ragebot_debug` - Toggle debug logging
- `/ragebot_vibecheck` - Show last 24h statistics
- `/ragebot_thresholds` - Show emoji thresholds
- `/ragebot_clear_old` - Delete old messages from database
- `/ragebot_die` - Stop the bot

## License

MIT
