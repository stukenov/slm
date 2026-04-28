"""Split sites between two servers for parallel crawling."""
import json

with open("sites_to_crawl.json") as f:
    sites = json.load(f)

half = len(sites) // 2
server1 = sites[:half]   # back
server2 = sites[half:]   # kaznu

with open("sites_server1.json", "w") as f:
    json.dump(server1, f, indent=2, ensure_ascii=False)
with open("sites_server2.json", "w") as f:
    json.dump(server2, f, indent=2, ensure_ascii=False)

print(f"Server 1 (back):  {len(server1)} sites")
print(f"Server 2 (kaznu): {len(server2)} sites")
