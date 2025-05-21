import json, re, pathlib, sys
from tqdm import tqdm

# Regular expressions for text processing
whaitespace_pattern  = re.compile(r"\s+")  # Matches all whitespace characters
bullet_pattern = re.compile(r"^[-‒•]\s*")  # Matches list markers at the start of a line
allowed_extantion = {".png", ".jpg", ".jpeg", ".webp"}  # Allowed image extensions

# Normalizes text by replacing non-breaking spaces with regular spaces
# removing extra spaces and trimming leading and trailing whitespace
def norm_text(txt):
    txt = txt.replace("\xa0", " ")
    txt = whaitespace_pattern.sub(" ", txt)
    return txt.strip()

# Normalizes a paragraph
# - Returns images as is
# - Normalizes text whitespace
def norm_paragraph(par):
    if par["type"] == "image":
        return par                      
    txt = norm_text(par["text"])
    if not txt:
        return None
    if par["type"] == "bullet":
        txt = bullet_pattern.sub("", txt)
        txt = "- " + txt
    return {"type": par["type"], "text": txt}

# Checks if the image has an allowed extension
# Returns True only for images with extensions .png, .jpg, .jpeg, .webp
def keep_image(img):
    ext = pathlib.Path(img.get("src", "")).suffix.lower()
    return ext in allowed_extantion

# Processes an article:
# 1. Normalizes all paragraphs
# 2. Filters images by allowed extensions
# 3. Updates image indices in paragraphs
# 4. Creates plain_text from text content
# 5. Removes "Andrew's letter" at the end of the article
def process_article(art):
    # 1. Normalize all paragraphs
    new_pars = []
    for p in art["paragraphs"]:
        np = norm_paragraph(p)
        if np:              
            new_pars.append(np)

    good_imgs = []
    old2new   = {}
    # 2. Filter images by allowed extensions
    for idx, img in enumerate(art.get("images", [])):
        if keep_image(img):
            old2new[idx] = len(good_imgs)
            good_imgs.append(img)

    # 3. Update image indices in paragraphs
    for p in new_pars:
        if p["type"] == "image":
            new_idx = old2new.get(p["index"])
            if new_idx is None:
                # If image was removed → convert alt text to plain text
                p.update({"type": "text", "text": f"[image removed]"})
            else:
                p["index"] = new_idx

    # 4. Create plain_text (only text and bullet points)
    plain_parts = []
    for p in new_pars:
        if p["type"] == "image":
            continue
        txt = p["text"]
        if txt.startswith(("Keep learning", "Keep building")):
            break                   # remove "Andrew's letter" at the end
        plain_parts.append(txt)

    art["paragraphs"] = new_pars
    art["images"]     = good_imgs
    art["plain_text"] = "\n".join(plain_parts) # join all text parts
    return art

# Collects all issue JSON files from the specified directory
# Returns a sorted list of file paths
def collect_issue_files(issues_root: pathlib.Path):
    return sorted(issues_root.rglob("issue-*.json"))

# Main function to process files:
# 1. Reads JSON files from input directory
# 2. Processes each article
# 3. Saves result in JSONL format
def main(in_root, out_root):
    in_root  = pathlib.Path(in_root)
    out_root = pathlib.Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    files = collect_issue_files(in_root)
    if not files:
        print("No issue-*.json files found", file=sys.stderr)
        sys.exit(1)

    for f in tqdm(files, desc="Cleaning"):
        issue_no = f.parent.name     
        with f.open(encoding="utf-8") as fp:
            raw_articles = json.load(fp)

        clean_articles = [process_article(a) for a in raw_articles]

        out_path = out_root / f"{issue_no}.jsonl"
        with out_path.open("w", encoding="utf-8") as out_fp:
            for art in clean_articles:
                json.dump(art, out_fp, ensure_ascii=False)
                out_fp.write("\n")

    print(f"Saved cleaned files → {out_root}")

# Start file processing
main("issues/", "data/clean/")
