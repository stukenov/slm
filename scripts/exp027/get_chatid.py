import urllib.request, json
url = "https://api.telegram.org/botREDACTED_TG_BOT_TOKEN/getUpdates"
resp = urllib.request.urlopen(url)
data = json.loads(resp.read())
for r in data.get("result", []):
    msg = r.get("message", {})
    chat = msg.get("chat", {})
    cid = chat.get("id")
    name = chat.get("first_name", "")
    text = msg.get("text", "")[:50]
    print(f"chat_id={cid} from={name} text={text}")
