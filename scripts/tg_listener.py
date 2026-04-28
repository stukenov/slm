#!/usr/bin/env python3
"""Simple Telegram bot polling — prints incoming messages to stdout."""
import urllib.request, urllib.parse, json, time, sys

TOKEN = "8620178354:AAFFqHqTvgobauCLiJ61CO1clWKG-CO-K1g"
CHAT_ID = "47474471"
BASE = f"https://api.telegram.org/bot{TOKEN}"
offset = 0

while True:
    try:
        url = f"{BASE}/getUpdates?offset={offset}&timeout=30"
        resp = urllib.request.urlopen(url, timeout=35)
        data = json.loads(resp.read())
        for upd in data.get("result", []):
            offset = upd["update_id"] + 1
            msg = upd.get("message", {})
            chat = msg.get("chat", {})
            if str(chat.get("id")) == CHAT_ID:
                text = msg.get("text", "")
                print(f"[TG] {text}", flush=True)
    except Exception as e:
        print(f"[TG-ERR] {e}", file=sys.stderr, flush=True)
        time.sleep(5)
