## Server Deploy Checklist (Linux)

### 1) Install service
```bash
sudo cp deploy/gemini.service /etc/systemd/system/gemini.service
sudo systemctl daemon-reload
sudo systemctl enable gemini
sudo systemctl start gemini
sudo systemctl status gemini
```

### 2) Environment file
Create `/etc/gemini/gemini.env`:
```env
# Optional alerts
DISCORD_WEBHOOK_URL=
TELEGRAM_BOT_TOKEN=
TELEGRAM_CHAT_ID=
HIGH_CONVICTION_THRESHOLD=0.80
ALERT_COOLDOWN_HOURS=6
```

### 3) Log rotation
```bash
sudo cp deploy/logrotate-gemini /etc/logrotate.d/gemini
sudo logrotate -f /etc/logrotate.d/gemini
```

### 4) Useful commands
```bash
journalctl -u gemini -f
systemctl restart gemini
systemctl stop gemini
```
