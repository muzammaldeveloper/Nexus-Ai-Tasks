
import os
import json
import time
import math
import re
import glob
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple

from dotenv import load_dotenv
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from tqdm import tqdm
import wikipedia
import requests
import PyPDF2
import ast

# Groq client & Chat integration (LangChain Groq wrapper)
try:
    from groq import Groq  # embeddings client
except Exception:
    Groq = None

try:
    from langchain_groq import ChatGroq  # chat model wrapper
except Exception:
    ChatGroq = None

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError("Please set GROQ_API_KEY in your .env file")

# ---------- Config ----------
PDF_FOLDER = "multi_tool_agent"
CHROMA_DB_DIR = "chroma_db"         # persists locally
CHROMA_COLLECTION_NAME = "GlobalMart.pdf"
EMBEDDING_MODEL = "nomic-embed-text-v1.5"   # example embedding model compatible with Groq / nomic
GROQ_CHAT_MODEL = "llama-3.1-8b-instant"  # choose a supported Groq chat model
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
TOP_K = 4   # retrieval top-k

# ---------- Utilities ----------
def safe_print(x):
    print(x, flush=True)

# ---------- PDF ingestion & chunking ----------
def load_pdfs_from_folder(folder: str) -> List[Tuple[str, str]]:
    """
    Return list of (filename, text) for all PDFs in folder.
    """
    files = glob.glob(os.path.join(folder, "*.pdf"))
    docs = []
    for f in files:
        try:
            with open(f, "rb") as fh:
                reader = PyPDF2.PdfReader(fh)
                texts = []
                for page in reader.pages:
                    try:
                        texts.append(page.extract_text() or "")
                    except Exception:
                        pass
                full_text = "\n".join(texts)
                docs.append((os.path.basename(f), full_text))
        except Exception as e:
            safe_print(f"[warn] failed to read PDF {f}: {e}")
    return docs

def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunk = text[start:end].strip()
        chunks.append(chunk)
        start = end - overlap
        if start < 0:
            start = 0
    return chunks

# ---------- Groq embeddings ----------
class GroqEmbeddingClient:
    def __init__(self, api_key: str):
        if Groq is None:
            raise RuntimeError("groq package not installed. pip install groq")
        self.client = Groq(api_key=api_key)

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Use Groq embeddings endpoint to embed a batch of texts.
        Returns list of vectors (floats).
        """
        # The groq client provides client.embeddings.create according to docs/examples.
        # We'll attempt to call it and extract embeddings - wrapper handles variations.
        embeddings = []
        B = 32
        for i in range(0, len(texts), B):
            batch = texts[i:i+B]
            resp = self.client.embeddings.create(
                input=batch,
                model=EMBEDDING_MODEL
            )
            # Response shape: resp.data is list of objects with 'embedding'
            # fallbacks handled defensively
            if hasattr(resp, "data"):
                for item in resp.data:
                    vec = item.embedding if hasattr(item, "embedding") else item.get("embedding")
                    embeddings.append(vec)
            else:
                # try dictionary access
                data = resp.get("data", [])
                for item in data:
                    vec = item.get("embedding")
                    embeddings.append(vec)
            time.sleep(0.1)
        return embeddings

# ---------- Chroma vector store helpers ----------
def create_chroma_client(persist_directory: str = CHROMA_DB_DIR):
    client = chromadb.Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory=persist_directory))
    return client

def ensure_collection(client, name: str):
    try:
        col = client.get_collection(name)
        return col
    except Exception:
        col = client.create_collection(name)
        return col

# ---------- Build or Refresh RAG DB ----------
def build_or_refresh_rag_db(api_key: str):
    safe_print("[info] Loading PDFs...")
    docs = load_pdfs_from_folder(PDF_FOLDER)
    if not docs:
        safe_print("[warn] No PDFs found in data/pdfs. Put your GlobalMart PDFs there and run again.")
        return None

    # chunk docs into text chunks and create metadatas
    texts = []
    metadatas = []
    ids = []
    for fname, fulltext in docs:
        chunks = chunk_text(fulltext)
        for idx, c in enumerate(chunks):
            texts.append(c)
            metadatas.append({"source": fname, "chunk": idx})
            ids.append(f"{fname}__{idx}")

    safe_print(f"[info] total chunks: {len(texts)}")

    # create Groq embeddings
    emb_client = GroqEmbeddingClient(api_key)
    safe_print("[info] Creating embeddings via Groq...")
    vectors = emb_client.embed_texts(texts)
    # convert to numpy arrays
    vectors = [np.array(v, dtype=np.float32) for v in vectors]

    # store in Chroma
    client = create_chroma_client()
    col = ensure_collection(client, CHROMA_COLLECTION_NAME)

    # clear existing collection to refresh (optional)
    try:
        client.delete_collection(CHROMA_COLLECTION_NAME)
        col = client.create_collection(CHROMA_COLLECTION_NAME)
    except Exception:
        # if delete not allowed, continue
        col = ensure_collection(client, CHROMA_COLLECTION_NAME)

    safe_print("[info] Upserting embeddings into Chroma...")
    # chroma client expects list of tuples
    col.add(
        ids=ids,
        documents=texts,
        metadatas=metadatas,
        embeddings=[v.tolist() for v in vectors]
    )
    client.persist()
    safe_print("[done] RAG DB built and persisted.")
    return True

# ---------- RAG retrieval and answer ----------
def rag_search_and_answer(query: str, groq_chat: Any, top_k: int = TOP_K) -> str:
    """
    1) retrieve top_k chunks from chroma
    2) call Groq LLM with system prompt + retrieved contexts + user query
    3) return generated answer
    """
    client = create_chroma_client()
    col = ensure_collection(client, CHROMA_COLLECTION_NAME)

    emb_client = GroqEmbeddingClient(GROQ_API_KEY)
    q_emb = emb_client.embed_texts([query])[0]

    # similarity search: use chroma query_by_vector
    results = col.query(query_embeddings=[q_emb], n_results=top_k, include=["documents", "metadatas", "distances"])
    docs = []
    for doc_list in results["documents"]:
        docs.extend(doc_list)

    # build context text
    context = ""
    for i, doc in enumerate(docs):
        context += f"Context {i+1}:\n{doc}\n\n"

    # construct prompt
    system_prompt = (
        "You are an expert product support assistant answering questions using the provided company documents. "
        "Use only the information in the provided contexts to answer, and cite the context numbers if relevant. "
        "If the answer isn't in the contexts, say so and provide a short suggestion."
    )
    user_prompt = f"User question: {query}\n\nCompany contexts:\n{context}\n\nAnswer concisely and clearly."

    # Call Groq chat (LangChain wrapper)
    if ChatGroq is None:
        return "[error] ChatGroq (langchain_groq) not installed."

    chat = ChatGroq(api_key=GROQ_API_KEY, model=GROQ_CHAT_MODEL, temperature=0.0, max_tokens=600)
    # Use a combined messages format
    resp = chat.invoke(f"{system_prompt}\n\n{user_prompt}")
    return resp.content.strip()

# ---------- Calculator tool (safe eval) ----------
def safe_eval_math(expr: str) -> str:
    """
    Safely evaluate arithmetic expressions using ast.
    Supports + - * / ** % // parentheses and numbers.
    """
    try:
        node = ast.parse(expr, mode="eval")

        # ensure nodes are safe
        for n in ast.walk(node):
            if isinstance(n, ast.Call):
                raise ValueError("Function calls not allowed")
            if isinstance(n, ast.Name):
                raise ValueError("Names not allowed")
        result = eval(compile(node, "<string>", mode="eval"), {"__builtins__": {}}, {})
        return str(result)
    except Exception as e:
        return f"[calc error] {e}"

# ---------- Wikipedia tool ----------
def wikipedia_search(query: str, sentences: int = 3) -> str:
    try:
        s = wikipedia.search(query, results=5)
        if not s:
            return f"No wikipedia results for '{query}'"
        # take first result summary
        title = s[0]
        summary = wikipedia.summary(title, sentences=sentences)
        return f"{title}: {summary}"
    except Exception as e:
        return f"[wikipedia error] {e}"

# ---------- Planner: ask Groq which tool to use ----------
def planner_decide_tools(user_question: str) -> Dict[str, Any]:
    """
    Use Groq Chat as a planner: it should return JSON indicating which tool to call and tool-input.
    We'll prompt it to respond with JSON like:
    {
      "tool": "RAG" | "CALCULATOR" | "WIKIPEDIA",
      "input": "text to pass to tool"
    }
    For multi-step, it may return a list of actions.
    """
    if ChatGroq is None:
        return {"error": "ChatGroq not installed"}

    planner_prompt = f"""
You are an agent planner. Given a user's question, decide which single tool is best to answer it.
Available tools:
- RAG: Use company documents (GlobalMart) - good for product-specific questions, specs, warranties, policies.
- CALCULATOR: use for math, currency conversion, totals, percentages, probability calculations.
- WIKIPEDIA: use for general knowledge questions about products/companies/history/definitions.

Return a JSON object with keys:
- actions: a list where each element is {{ "tool": TOOL_NAME, "input": "text for that tool" }}
Pick 1 to 3 actions in logical order. Keep JSON valid.

User question:
\"\"\"{user_question}\"\"\"
"""

    chat = ChatGroq(api_key=GROQ_API_KEY, model=GROQ_CHAT_MODEL, temperature=0.0, max_tokens=400)
    resp = chat.invoke(planner_prompt)
    text = resp.content.strip()


    try:
       
        json_text_match = re.search(r"(\{[\s\S]*\})", text)
        if json_text_match:
            json_text = json_text_match.group(1)
        else:
            json_text = text
        result = json.loads(json_text)
        return result
    except Exception as e:
        # Fallback: try heuristic simple routing
        low = user_question.lower()
        if any(w in low for w in ("how much", "convert", "price", "cost", "calculate", "total", "sum", "percent")):
            return {"actions": [{"tool": "CALCULATOR", "input": user_question}]}
        if any(w in low for w in ("what is", "who is", "history", "when was")):
            return {"actions": [{"tool": "WIKIPEDIA", "input": user_question}]}
        # default
        return {"actions": [{"tool": "RAG", "input": user_question}]}

# ---------- Orchestrator ----------
def run_agent(user_question: str) -> str:
    safe_print(f"[user] {user_question}")
    plan = planner_decide_tools(user_question)
    if "error" in plan:
        return "[error] planner missing"

    actions = plan.get("actions") or plan.get("action") or []
    final_parts = []
    for act in actions:
        tool = act.get("tool", "").upper()
        inp = act.get("input", "")
        safe_print(f"[plan] tool={tool} input={inp}")

        if tool == "RAG":
            out = rag_search_and_answer(inp, groq_chat=None)
            final_parts.append(f"[RAG Answer]\n{out}")
        elif tool == "CALCULATOR":
            
            expr = inp
           
            out = safe_eval_math(expr)
            final_parts.append(f"[Calculator]\n{out}")
        elif tool == "WIKIPEDIA":
            out = wikipedia_search(inp, sentences=3)
            final_parts.append(f"[Wikipedia]\n{out}")
        else:
            final_parts.append(f"[unknown tool {tool}]")

    
    synthesis_prompt = (
        "You are an assistant that must synthesize the outputs from multiple tools into one final helpful answer.\n\n"
        "Tool outputs:\n" + "\n\n".join(final_parts) + "\n\n"
        "Write a concise final answer for the user that integrates the results above and mentions which tool provided which info."
    )
    if ChatGroq is None:
        return "\n\n".join(final_parts)

    chat = ChatGroq(api_key=GROQ_API_KEY, model=GROQ_CHAT_MODEL, temperature=0.2, max_tokens=400)
    resp = chat.invoke(synthesis_prompt)
    return resp.content.strip()

# ---------- Simple interactive CLI ----------
def interactive_cli():
    safe_print("Multi-Tool Agent (RAG + Calculator + Wikipedia) — type 'exit' to quit.")
    while True:
        q = input("\nYour question > ").strip()
        if not q:
            continue
        if q.lower() in ("exit", "quit"):
            break
        ans = run_agent(q)
        print("\n=== AGENT ANSWER ===\n")
        print(ans)
        print("\n=== END ===\n")

# ---------- Entrypoint ----------
def main():
    
    if not os.path.exists(CHROMA_DB_DIR) or not os.listdir(CHROMA_DB_DIR):
        safe_print("[info] Building RAG DB from PDFs...")
        build_or_refresh_rag_db(GROQ_API_KEY)
    else:
        safe_print("[info] Existing Chroma DB detected - using it. To rebuild, delete chroma_db/ and rerun.")

    # start CLI
    interactive_cli()

if __name__ == "__main__":
    main()










