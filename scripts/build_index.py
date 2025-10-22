import os, glob, pickle
from pathlib import Path
from typing import Dict, Any, List
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

DATA_DIR = os.getenv("DATA_DIR", "data")
INDEX_DIR = os.getenv("INDEX_DIR", "index")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "600"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "120"))
EMB_MODEL = os.getenv("EMB_MODEL_NAME", "BAAI/bge-m3")
EMB_BATCH = int(os.getenv("EMB_BATCH", "32"))  

def _as_dict(md: Any) -> Dict[str, Any]:
    """Her ihtimale karşı metadata'yı dict'e zorla."""
    if md is None:
        return {}
    if isinstance(md, dict):
        return md
    return {"source": str(md)}

def detect_course_from_path(p: str) -> str:
    """
    data/<course>/... yolundaki <course> klasörünü ders adı olarak kullan.
    Örn: data/ceng_102/week3.pdf -> 'ceng_102'
    """
    try:
        parts = Path(p).parts
        if "data" in parts:
            i = parts.index("data")
            if i + 1 < len(parts):
                return parts[i + 1].lower()
    except Exception:
        pass
    return "unknown"

def detect_unit_from_path(p: str) -> str:
    """
    data/<course>/<unit>/... gibi bir yapı varsa unit'i de yakala (opsiyonel).
    Örn: data/CENG102/PLC_204/lec1.pdf -> 'PLC_204'
    """
    try:
        parts = list(Path(p).parts)
        if "data" in parts:
            i = parts.index("data")
            
            if i + 2 < len(parts):
                return parts[i + 2]
    except Exception:
        pass
    return ""

def load_documents() -> List:
    docs = []
    pdfs = glob.glob(os.path.join(DATA_DIR, "**/*.pdf"), recursive=True)
    txts = glob.glob(os.path.join(DATA_DIR, "**/*.txt"), recursive=True)
    mds  = glob.glob(os.path.join(DATA_DIR, "**/*.md"),  recursive=True)

    # PDF -> sayfa sayfa Document
    for p in pdfs:
        ld = PyPDFLoader(p).load()
        for d in ld:
            md = _as_dict(getattr(d, "metadata", {}))
            md.setdefault("source", os.path.basename(p))
            md["page"] = md.get("page", md.get("page_number"))
            md.setdefault("course", detect_course_from_path(p))
            unit = md.get("unit") or detect_unit_from_path(p)
            if unit:
                md["unit"] = unit
            md.setdefault("path", str(Path(p)))
            d.metadata = md
        docs += ld

    # TXT & MD -> tek Document
    for p in txts + mds:
        ld = TextLoader(p, encoding="utf-8").load()
        for d in ld:
            md = _as_dict(getattr(d, "metadata", {}))
            md.setdefault("source", os.path.basename(p))
            md.setdefault("course", detect_course_from_path(p))
            unit = md.get("unit") or detect_unit_from_path(p)
            if unit:
                md["unit"] = unit
            md.setdefault("path", str(Path(p)))
            md["page"] = md.get("page", None)
            d.metadata = md
        docs += ld

    if not docs:
        raise SystemExit("[!] data/ boş. Lütfen ders PDF/TXT/MD dosyalarını ekleyin.")
    return docs

def main(rebuild: bool = False):
    os.makedirs(INDEX_DIR, exist_ok=True)
    faiss_path  = os.path.join(INDEX_DIR, "faiss.index")
    chunks_path = os.path.join(INDEX_DIR, "chunks.pkl")

    if (os.path.exists(faiss_path) or os.path.exists(chunks_path)) and not rebuild:
        raise SystemExit("[!] index/ zaten var. Üzerine yazmak için --rebuild kullan.")

    docs = load_documents()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        is_separator_regex=False,
    )
    chunks = splitter.split_documents(docs)

    for c in chunks:
        c.metadata = _as_dict(getattr(c, "metadata", {}))
        c.metadata.setdefault("source", c.metadata.get("source", "unknown"))
        c.metadata.setdefault("course", c.metadata.get("course", "unknown"))
        c.metadata.setdefault("path", c.metadata.get("path", ""))

    texts = [c.page_content for c in chunks]
    print(f"[*] Chunk sayısı: {len(texts)}")

    print("[*] Embedding hesaplanıyor…")
    emb = SentenceTransformer(EMB_MODEL)
    X = emb.encode(
        texts,
        batch_size=EMB_BATCH,
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    X = np.asarray(X, dtype="float32")

    print("[*] FAISS yazılıyor…")
    index = faiss.IndexFlatIP(X.shape[1])  
    index.add(X)
    faiss.write_index(index, faiss_path)

    with open(chunks_path, "wb") as f:
        pickle.dump(chunks, f)

    sample_md = type(chunks[0].metadata).__name__
    print(f"[✓] Kaydedildi → {faiss_path}  |  {chunks_path} (metadata type: {sample_md})")

if __name__ == "__main__":
    import sys
    main(rebuild="--rebuild" in sys.argv)
