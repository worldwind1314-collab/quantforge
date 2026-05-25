#!/bin/bash
# QuantForge health monitor — check API, DB, and data freshness
# Run via cron every 30 minutes: */30 * * * * /var/www/quantforge/scripts/health_monitor.sh

set -e

API_URL="${API_URL:-http://localhost:8000}"
ALERT_WEBHOOK="${ALERT_WEBHOOK:-}"
LOG_FILE="${LOG_FILE:-/var/log/quantforge-health.log}"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"; }

alert() {
    msg="[QuantForge ALERT] $1"
    log "ALERT: $msg"
    if [ -n "$ALERT_WEBHOOK" ]; then
        curl -s -X POST -H "Content-Type: application/json" \
            -d "{\"text\":\"$msg\"}" "$ALERT_WEBHOOK" > /dev/null 2>&1 || true
    fi
}

# ── 1. API liveness ──
HTTP_CODE=$(curl -s -o /dev/null -w '%{http_code}' --max-time 10 "$API_URL/api/health" 2>/dev/null || echo "000")
if [ "$HTTP_CODE" != "200" ]; then
    alert "API health check failed: HTTP $HTTP_CODE"
    exit 1
fi

# ── 2. Data freshness ──
FRESHNESS=$(curl -s --max-time 10 "$API_URL/api/market/pipeline/status" 2>/dev/null || echo '{}')
LATEST_DATE=$(echo "$FRESHNESS" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('latest_quote_date',''))" 2>/dev/null || echo "")

if [ -z "$LATEST_DATE" ]; then
    alert "Cannot determine latest data date"
else
    DAYS_OLD=$(( ($(date +%s) - $(date -d "$LATEST_DATE" +%s 2>/dev/null || date +%s)) / 86400 ))
    if [ "$DAYS_OLD" -gt 3 ]; then
        alert "Data is $DAYS_OLD days stale (latest: $LATEST_DATE)"
    fi
fi

# ── 3. Disk space ──
DISK_PCT=$(df /var | awk 'NR==2 {print $5}' | tr -d '%')
if [ "${DISK_PCT:-0}" -gt 85 ]; then
    alert "Disk usage at ${DISK_PCT}% on /var"
fi

# ── 4. Memory ──
MEM_PCT=$(free | awk '/Mem:/ {printf "%.0f", $3/$2*100}')
if [ "${MEM_PCT:-0}" -gt 90 ]; then
    alert "Memory usage at ${MEM_PCT}%"
fi

log "Health check passed (data: $LATEST_DATE, disk: ${DISK_PCT:-?}%, mem: ${MEM_PCT:-?}%)"
