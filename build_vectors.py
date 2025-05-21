import json, glob, pathlib
import torch
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_experimental.open_clip import OpenCLIPEmbeddings
from langchain_chroma import Chroma

# Configuration for file paths and directories
dir_glob = "data/clean/issue-*.jsonl"  # Pattern to find all cleaned issue files
index_dir = pathlib.Path("index")      # Directory to store vector embeddings
index_dir.mkdir(exist_ok=True, parents=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Initialize text embedding model
txt_embed = HuggingFaceEmbeddings(model_name="intfloat/e5-base-v2", model_kwargs = {"device": device})

# Initialize image embedding model
img_embed = OpenCLIPEmbeddings(model_name = "ViT-B-32", checkpoint = "openai", device = device)

# Splits text into chunks of 450 characters with 100 character overlap
splitter = RecursiveCharacterTextSplitter(chunk_size=450, chunk_overlap=100)

# Lists to store document objects for text and images
text_docs, image_docs = [], []

# Process each issue file
for path in glob.glob(dir_glob):
    issue_num = int(pathlib.Path(path).stem.split("-")[1])
    for art in map(json.loads, open(path, encoding="utf-8")):     # Process each article in the issue
        for pos, chunk in enumerate(splitter.split_text(art["plain_text"])):         # Split article text into chunks and create text documents
            text_docs.append(
                Document(page_content=chunk,
                         metadata={"kind":"chunk",
                                   "issue_no":   issue_num,
                                   "article_id":art["id"],
                                   "title":art["title"],
                                   "pos": pos})
            )      

        for img in art.get("images", []):         # Create image documents for each image in the article
            image_docs.append(
                Document(page_content=img["path"],
                         metadata={"kind":"image",
                                   "issue_no": issue_num,
                                   "article_id":art["id"],
                                   "title":art["title"],
                                   "alt":img["alt"],
                                   "src": img["src"]})
            )

print(f"{len(text_docs)} text chunks | {len(image_docs)} images")

# Create and store vector embeddings in Chroma database
# Separate collections for text and images
Chroma.from_documents(text_docs,  txt_embed, collection_name="text", persist_directory=str(index_dir/"text"))
Chroma.from_documents(image_docs, img_embed, collection_name="image", persist_directory=str(index_dir/"image"))
print(f"Embeddings stored in Chroma using {device}")


