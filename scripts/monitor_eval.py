#!/usr/bin/env python3
"""Monitor eval progress on kaznu and send Telegram updates every 15 min."""

import json
import subprocess
import time
import urllib.request
import sys
from datetime import datetime

TOKEN = "5159241157:AAGksR3Dm_5DwxHZStjC2mNq7Z3iNZOxO68"
CHAT_ID = 47474471
INTERVAL = 900  # 15 min
SSH = ["ssh", "-o", "ConnectTimeout=180", "-o", "ServerAliveInterval=15", "-p", "15126", "root@164.138.46.36"]


def send_tg(text):
    try:
        data = json.dumps({"chat_id": CHAT_ID, "text": text}).encode()
        req = urllib.request.Request(
            f"https://api.telegram.org/bot{TOKEN}/sendMessage",
            data=data, headers={"Content-Type": "application/json"},
        )
        urllib.request.urlopen(req, timeout=30)
    except Exception as e:
        print(f"TG error: {e}", flush=True)


def ssh_cmd(cmd, timeout=200):
    try:
        r = subprocess.run(SSH + [cmd], capture_output=True, text=True, timeout=timeout)
        return r.stdout.strip() if r.returncode == 0 else None
    except Exception:
        return None


def check():
    # Count result files
    n = ssh_cmd('find /root/slm/paper/results -name "*.json" ! -name "summary*" ! -name "._*" 2>/dev/null | wc -l')
    if n is None:
        return None

    n = int(n.strip())

    # Screen alive?
    scr = ssh_cmd('screen -ls 2>/dev/null | grep -c eval_all || echo 0')
    screen_alive = scr and scr.strip() != "0"

    # Last meaningful log line
    last = ssh_cmd('grep -E "^(---|  \\[|===)" /root/slm/logs/eval_all.log 2>/dev/null | tail -3')

    return {"files": n, "total": 84, "screen": screen_alive, "last": (last or "")[:200]}


def main():
    print(f"Monitor started at {datetime.now():%H:%M}", flush=True)

    while True:
        p = check()
        now = datetime.now().strftime("%H:%M")

        if p is None:
            send_tg(f"⚠️ [{now}] kaznu unreachable")
            print(f"[{now}] unreachable", flush=True)
        else:
            pct = int(p["files"] / p["total"] * 100)
            bar = "█" * (pct // 10) + "░" * (10 - pct // 10)
            status = "🟢" if p["screen"] else "🔴 STOPPED"

            msg = f"📊 [{now}] {bar} {pct}% ({p['files']}/{p['total']})\n{status}\n"
            if p["last"]:
                msg += f"\n{p['last']}"

            send_tg(msg)
            print(f"[{now}] {p['files']}/{p['total']}", flush=True)

            if p["files"] >= p["total"]:
                send_tg("✅ Eval COMPLETE! Все 84 результата собраны.")
                print("DONE!", flush=True)
                sys.exit(0)

            if not p["screen"] and p["files"] < p["total"]:
                send_tg(f"🔴 Screen упал на {p['files']}/{p['total']}. Проверь логи.")
                print("Screen died!", flush=True)
                sys.exit(1)

        time.sleep(INTERVAL)


if __name__ == "__main__":
    main()
