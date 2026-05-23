# QuantForge 熔量

AI-powered quantitative trading platform for A-share market.

## Stack

- **Backend**: FastAPI + SQLAlchemy + PostgreSQL
- **Data**: AKShare (A-share market data)
- **AI**: DeepSeek API (financial NLP)
- **Deploy**: Alibaba Cloud + systemd + GitHub Actions

## Project Structure

```
quantforge/
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI entry point
│   │   ├── core/                 # Config, database
│   │   ├── models/               # SQLAlchemy models
│   │   ├── services/             # Data pipeline, AI
│   │   └── api/                  # REST endpoints
│   └── requirements.txt
└── scripts/
    └── sync_data.py              # Standalone data sync
```

## Quick Start

```bash
cd backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # edit as needed
uvicorn app.main:app --reload
```

## Server Setup

```bash
# Create systemd service
cat > /etc/systemd/system/quantforge-api.service << 'SERVICE'
[Unit]
Description=QuantForge API
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/var/www/quantforge/backend
ExecStart=/var/www/quantforge/backend/venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8001
Restart=always

[Install]
WantedBy=multi-user.target
SERVICE

systemctl daemon-reload
systemctl enable --now quantforge-api

# Create database
sudo -u postgres psql -c "CREATE DATABASE quantforge;"

# Initial data sync
cd /var/www/quantforge
venv/bin/python scripts/sync_data.py
```
