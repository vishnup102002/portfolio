import re
import urllib.request
import os
from urllib.parse import urlparse

os.makedirs('assets/icons', exist_ok=True)

with open('index.html', 'r') as f:
    content = f.read()

# Find all image sources in the skills section
# We can just find all src="http..." in the file since most of them are in skills.
urls = re.findall(r'src="(https?://[^"]+)"', content)

for url in set(urls):
    filename = os.path.basename(urlparse(url).path)
    if not filename:
        filename = "icon.png"
    # Some urls have query params or weird names
    if "79963867" in url: filename = "qdrant.png"
    if "37254256" in url: filename = "weaviate.png"
    if "108502392" in url: filename = "chromadb.png"
    if "generative-ai" in url: filename = "gen-ai.jpg"
    if "je-yj93ERqG0BEoQ" in url: filename = "langgraph.png"
    
    local_path = os.path.join('assets/icons', filename)
    try:
        urllib.request.urlretrieve(url, local_path)
        content = content.replace(url, local_path)
        print(f"Downloaded {url} to {local_path}")
    except Exception as e:
        print(f"Failed to download {url}: {e}")

with open('index.html', 'w') as f:
    f.write(content)

