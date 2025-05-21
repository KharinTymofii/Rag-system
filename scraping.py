import os, re, json, time, requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from pathlib import Path
from tqdm import tqdm

base_url = "https://www.deeplearning.ai/the-batch/issue-{num}/"

#  Converts text into a URL-friendly slug format
def slug(text):
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")

# Extracts the file extension from a URL
# If no extension is found, defaults to .jpg
def file_ext(url):
    ext = os.path.splitext(urlparse(url).path)[1].lower()
    return ext if ext else ".jpg"

#Extracts the first image source and alt text from a figure element
# Handles both regular src and srcset attributes
def first_img(fig):
    img = fig.find("img")
    if not img:
        return None, None
    src = img.get("src")
    if img.has_attr("srcset"):
        src = img["srcset"].split(",")[-1].split()[0]
    return src, img.get("alt", "")

# Downloads an image from a URL and saves it to a specified folder with a spicific naming convention
def download_img(url, folder, issue_num=0, slug="", idx=0):
    if not url:
        return None
    try:
        folder.mkdir(parents=True, exist_ok=True)
        image_name = f"issue_{issue_num}_{slug}_{idx}{file_ext(url)}"
        path = folder / image_name
        data = requests.get(url).content
        with open(path, 'wb') as f:
            f.write(data)
        return str(path)
    except Exception as e:
        tqdm.write(f"Failed to download {url}: {e}")
        return None

# Parses a specific issue of The Batch newsletter and extracts all content
# This function:
# 1. Downloads the webpage for a specific issue number
# 2. Finds the "News" section in the HTML
# 3. Processes the introduction section:
#    - Extracts text paragraphs
#    - Downloads and saves images
#    - Handles bullet points and subtitles
# 4. Processes each article in the news section:
#    - Gets article title and ID
#    - Extracts all paragraphs and text content
#    - Downloads and saves associated images
#    - Handles bullet points and formatting
# 5. Saves everything as a structured JSON file with:
#    - All content is saved in the 'issues/issue-{num}/' directory
def parse_issue(issue_num: int):
    url = base_url.format(num=issue_num)
    try:
        html = requests.get(url, timeout=20)
        if html.status_code == 404:
            print(f"Issue {issue_num} not found")
            return
        soup = BeautifulSoup(html.text, 'lxml')
    except Exception as e:
        print(f"Issue {issue_num} failed to load: {e}")
        return

    news_start = soup.find("h1", id="news")
    if not news_start:
        print(f"Issue {issue_num} has no 'News' section")
        return

    articles = []
    base_dir = Path("issues") / f"issue-{issue_num}"
    image_dir = base_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    intro_pars = []
    intro_imgs = []
    pending_fig = None

    # Process the introduction section
    for prev in reversed(list(news_start.previous_siblings)):
        name = getattr(prev, "name", None)
        if not name:
            continue
        if name == "figure":
            src, alt = first_img(prev)
            if src:
                idx = len(intro_imgs)
                path = download_img(src, image_dir, issue_num, "intro", idx)
                intro_imgs.append({"path": path, "src": src, "alt": alt})
                intro_pars.append({"type": "image", "index": idx})
        elif name == "p":
            intro_pars.append({"type": "text", "text": prev.get_text(" ", strip=True)})
        elif name == "ul":
            for li in prev.find_all("li", recursive=False):
                intro_pars.append({"type": "bullet", "text": li.get_text(" ", strip=True)})
        elif name == "li":
            intro_pars.append({"type": "bullet", "text": prev.get_text(" ", strip=True)})
        elif name == "h2":
            intro_pars.append({"type": "subtitle", "text": prev.get_text(" ", strip=True)})

    if intro_pars or intro_imgs:
        articles.append({
            "id": "intro",
            "title": "Issue Introduction",
            "images": intro_imgs,
            "paragraphs": intro_pars
        })

    # Process the articles in the news section
    for node in news_start.find_all_next():
        tag = getattr(node, "name", None)
        if tag == "figure":
            pending_fig = node
            continue
        if tag in {"h1", "h2"} and node.get("id") and node.get("id") != "news":
            art_id = node["id"]
            title = node.get_text(strip=True)
            art_imgs = []
            art_pars = []

            if pending_fig:
                src, alt = first_img(pending_fig)
                if src:
                    idx = len(art_imgs)
                    path = download_img(src, image_dir, issue_num, art_id, idx)
                    art_imgs.append({"path": path, "src": src, "alt": alt})
                    art_pars.append({"type": "image", "index": idx})
                pending_fig = None

            for sib in node.next_siblings:
                sib_name = getattr(sib, "name", None)
                if sib_name in {"h1", "hr"}:
                    break
                if sib_name == "figure":
                    src, alt = first_img(sib)
                    if src:
                        idx = len(art_imgs)
                        path = download_img(src, image_dir, issue_num, art_id, idx)
                        art_imgs.append({"path": path, "alt": alt})
                        art_pars.append({"type": "image", "index": idx})
                elif sib_name == "p":
                    art_pars.append({"type": "text", "text": sib.get_text(" ", strip=True)})
                elif sib_name == "ul":
                    for li in sib.find_all("li"):
                        art_pars.append({"type": "bullet", "text": li.get_text(" ", strip=True)})
                elif sib_name == "li":
                    art_pars.append({"type": "bullet", "text": sib.get_text(" ", strip=True)})

            articles.append({
                "id": art_id,
                "title": title,
                "images": art_imgs,
                "paragraphs": art_pars
            })

    # Save the issue content to a JSON file
    json_path = base_dir / f"issue-{issue_num}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(articles, f, ensure_ascii=False, indent=2)

    print(f"Issue {issue_num} saved to {json_path}")

# Initializes the script to parse issues from 300 to 284
for num in tqdm(range(300, 283, -1), desc="Batch issues"):
    parse_issue(num)
    time.sleep(1.5) 
