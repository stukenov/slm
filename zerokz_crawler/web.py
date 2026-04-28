"""Web UI for kaznet crawler monitoring."""

import json
import os
import subprocess
from pathlib import Path

from flask import Flask, render_template_string

app = Flask(__name__)
BASE = Path(__file__).parent

HTML = """
<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Kaznet Crawler</title>
<meta http-equiv="refresh" content="30">
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: -apple-system, sans-serif; background: #0f1117; color: #e1e4e8; padding: 24px; }
  h1 { font-size: 1.5rem; margin-bottom: 20px; color: #58a6ff; }
  .stats { display: flex; gap: 16px; flex-wrap: wrap; margin-bottom: 24px; }
  .stat { background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 16px 24px; min-width: 140px; }
  .stat .value { font-size: 2rem; font-weight: 700; color: #f0f6fc; }
  .stat .label { font-size: 0.85rem; color: #8b949e; margin-top: 4px; }
  .bar-wrap { background: #21262d; border-radius: 6px; height: 24px; margin-bottom: 24px; overflow: hidden; }
  .bar { background: linear-gradient(90deg, #238636, #2ea043); height: 100%; border-radius: 6px; transition: width 0.3s; display: flex; align-items: center; justify-content: center; font-size: 0.8rem; font-weight: 600; }
  .section { margin-bottom: 24px; }
  .section h2 { font-size: 1.1rem; color: #8b949e; margin-bottom: 12px; }
  table { width: 100%; border-collapse: collapse; }
  th, td { text-align: left; padding: 8px 12px; border-bottom: 1px solid #21262d; font-size: 0.85rem; }
  th { color: #8b949e; }
  .status-on { color: #3fb950; }
  .status-off { color: #f85149; }
  .btn { display: inline-block; padding: 8px 16px; border-radius: 6px; border: 1px solid #30363d; background: #21262d; color: #c9d1d9; cursor: pointer; font-size: 0.85rem; text-decoration: none; margin-right: 8px; }
  .btn:hover { background: #30363d; }
  .refresh { float: right; font-size: 0.8rem; }
  .scrollable { max-height: 500px; overflow-y: auto; }
  .tag { display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 0.75rem; }
  .tag-ok { background: #238636; color: #fff; }
  .tag-pending { background: #30363d; color: #8b949e; }
</style>
</head>
<body>
<h1>Kaznet Crawler <span class="refresh"><a href="/" class="btn">Refresh</a></span></h1>

<div class="stats">
  <div class="stat">
    <div class="value">{{ sites_total }}</div>
    <div class="label">Sites total</div>
  </div>
  <div class="stat">
    <div class="value">{{ sites_done }}</div>
    <div class="label">Sites crawled</div>
  </div>
  <div class="stat">
    <div class="value">{{ total_pages }}</div>
    <div class="label">Pages saved</div>
  </div>
  <div class="stat">
    <div class="value">{{ disk_mb }} MB</div>
    <div class="label">Disk used</div>
  </div>
  <div class="stat">
    <div class="value">{{ uploaded }}</div>
    <div class="label">Uploaded to HF</div>
  </div>
  <div class="stat">
    <div class="value {% if crawler_running %}status-on{% else %}status-off{% endif %}">{{ "ON" if crawler_running else "OFF" }}</div>
    <div class="label">Crawler</div>
  </div>
</div>

<div class="bar-wrap">
  <div class="bar" style="width: {{ pct }}%">{{ pct }}%</div>
</div>

<div class="section">
  <h2>Sites ({{ sites_done }}/{{ sites_total }})</h2>
  <div class="scrollable">
  <table>
    <tr><th>#</th><th>Domain</th><th>Name</th><th>Pages</th><th>Status</th></tr>
    {% for s in sites %}
    <tr>
      <td>{{ loop.index }}</td>
      <td>{{ s.domain }}</td>
      <td>{{ s.name }}</td>
      <td>{{ s.pages }}</td>
      <td><span class="tag {% if s.done %}tag-ok{% else %}tag-pending{% endif %}">{{ "done" if s.done else "pending" }}</span></td>
    </tr>
    {% endfor %}
  </table>
  </div>
</div>
</body>
</html>
"""


def get_stats():
    # Sites catalog
    sites_file = BASE / "sites_to_crawl.json"
    sites_list = []
    if sites_file.exists():
        with open(sites_file) as f:
            sites_list = json.load(f)

    # Crawl progress
    progress_file = BASE / "crawl_sites_progress.json"
    done_domains = set()
    if progress_file.exists():
        with open(progress_file) as f:
            done_domains = set(json.load(f))

    # Upload progress
    upload_file = BASE / "upload_log.json"
    uploaded = 0
    if upload_file.exists():
        with open(upload_file) as f:
            uploaded = len(json.load(f).get("uploaded_domains", []))

    # Count pages per site
    crawled_dir = BASE / "crawled_sites"
    total_pages = 0
    disk_bytes = 0
    site_pages = {}
    if crawled_dir.exists():
        for d in crawled_dir.iterdir():
            if d.is_dir():
                pages = list(d.glob("*.html"))
                site_pages[d.name] = len(pages)
                for p in pages:
                    disk_bytes += p.stat().st_size
                total_pages += len(pages)

    # Build sites table
    from urllib.parse import urlparse
    sites_display = []
    for s in sites_list:
        domain = s.get("domain", "")
        dirname = domain.replace(":", "_").replace("/", "_").replace(".", "_")
        sites_display.append({
            "domain": domain,
            "name": s.get("name", "")[:40],
            "pages": site_pages.get(dirname, 0),
            "done": domain in done_domains,
        })
    # Sort: done first (by pages desc), then pending
    sites_display.sort(key=lambda x: (-x["done"], -x["pages"]))

    # Crawler running?
    try:
        r = subprocess.run(["pgrep", "-f", "crawl_all"], capture_output=True, timeout=5)
        crawler_running = r.returncode == 0
    except Exception:
        crawler_running = False

    sites_total = len(sites_list)
    sites_done = len(done_domains)
    pct = round(sites_done / sites_total * 100, 1) if sites_total > 0 else 0

    return {
        "sites_total": sites_total,
        "sites_done": sites_done,
        "total_pages": total_pages,
        "disk_mb": round(disk_bytes / 1024 / 1024, 1),
        "uploaded": uploaded,
        "pct": pct,
        "crawler_running": crawler_running,
        "sites": sites_display,
    }


@app.route("/")
def index():
    return render_template_string(HTML, **get_stats())


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8585)
