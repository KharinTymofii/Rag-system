from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_experimental.open_clip import OpenCLIPEmbeddings
from dotenv import load_dotenv
import gradio as gr
load_dotenv()

# Load the ve ctor stores for text and images
def load_vector_stores():
    txt_embed = HuggingFaceEmbeddings(model_name="intfloat/e5-base-v2", model_kwargs={"device": "cuda"})
    img_embed = OpenCLIPEmbeddings(model_name="ViT-B-32", checkpoint="openai", device="cuda")

    text_vs  = Chroma(persist_directory=str("index/text"), collection_name="text", embedding_function=txt_embed)
    image_vs = Chroma(persist_directory=str("index/image"), collection_name="image", embedding_function=img_embed)
    return text_vs, image_vs

# Builds context for the question by finding relevant text and images
def build_context(question, text_vs, image_vs):
    # Get relevant text chunks (unique only)
    texts  = text_vs.similarity_search(question, k=8)
    unique = []
    seen = set()
    for d in texts:
        s = d.page_content
        if s not in seen:
            unique.append(d)
            seen.add(s)
    texts = unique[:4]         

    # Get relevant images
    images = image_vs.similarity_search(question, k=2)

    # Build context blocks
    blocks = []
    for i, d in enumerate(texts):
        src = d.metadata.get("title", "unknown-article")
        blocks.append({
            "type": "text",
            "text": f"[Text {i} | {src}] {d.page_content}"
        })

    for i, d in enumerate(images): 
        url = d.metadata["src"]    
        alt = d.metadata["alt"]
        art  = d.metadata.get("title", "unknown-article")

        blocks.append({                    
            "type": "image_url",
            "image_url": {"url": url}
        })
        blocks.append({                 
            "type": "text",
            "text": f"[Image {i} | {art}] {alt} {{image:{i}}}"
        })
    return blocks

SYSTEM_PROMPT = (
    "You are an expert AI news assistant with access to The Batch newsletter archives. "
    "Your task is to provide accurate, informative answers based on the provided context. "
    "Guidelines:"
    "1. Use ONLY the provided text and image captions as your source of information"
    "2. If the context doesn't contain relevant information, say so"
    "3. You will receive image URLs, describe if needed what you see in the images based on their descriptions"
    "4. Keep answers concise (≤200 words) but comprehensive"
    "5. When referencing images, use {{image:N}} format"
    "6. Maintain a professional and informative tone"
    "7. If multiple sources provide different information, mention this"
    "8. Focus on factual information from the provided context"
    "9. Avoid making assumptions beyond the given context"
)

# Initialize vector stores and LLM
text_vs, image_vs = load_vector_stores()
llm = ChatOpenAI(model="gpt-4o", temperature=0)

# Creates a markdown list of clickable links to unique articles used
def collect_sources(txt_docs, img_docs):
    seen = {}
    for d in txt_docs + img_docs:
        issue = d.metadata.get("issue_no")
        aid   = d.metadata.get("article_id")
        title = d.metadata.get("title", "untitled")
        if (issue, aid) not in seen:
            seen[(issue, aid)] = title

    md_lines = [
        f"- [Issue {issue} — {title}](https://www.deeplearning.ai/the-batch/issue-{issue}/#{aid})"
        for (issue, aid), title in seen.items()
    ]
    return "\n".join(md_lines) if md_lines else "- (none)"

# Function to handle the question and generate an answer
def rag_answer(q):
    blocks  = build_context(q, text_vs, image_vs)
    blocks.append({"type":"text", "text": q})

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": blocks}
    ]

    reply = llm.invoke(messages).content
    img_docs = image_vs.similarity_search(q, k=2)
    imgs     = [d.metadata["src"] for d in img_docs]

    txt_docs = text_vs.similarity_search(q, k=4)
    sources_md = collect_sources(txt_docs, img_docs)
    return reply, imgs, sources_md


# Create Gradio interface
with gr.Blocks(title="The-Batch Multimodal RAG") as demo:
    gr.Markdown("## Ask about AI news")
    inp = gr.Textbox(label="Your question", placeholder="What is Andrew Ng's view on AI regulation?")
    out_answer = gr.Markdown()
    out_gallery = gr.Gallery(label="Images", columns=2, height="auto")
    out_sources = gr.Markdown(label="Articles used")
    btn = gr.Button("Search")
    btn.click(rag_answer, inp, [out_answer, out_gallery, out_sources])

if __name__ == "__main__":
    demo.launch()





