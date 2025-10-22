# rag_pipeline.py — FIXED

import os, pickle, re
from typing import List, Dict, Tuple, Optional
import numpy as np

from sentence_transformers import SentenceTransformer
import faiss
import google.generativeai as genai

# ====== ENV / PATHS ======
EMB_MODEL_NAME = os.getenv("EMB_MODEL_NAME", "BAAI/bge-m3")
INDEX_DIR      = os.getenv("INDEX_DIR", "index")
FAISS_PATH     = os.path.join(INDEX_DIR, "faiss.index")
CHUNKS_PATH    = os.path.join(INDEX_DIR, "chunks.pkl")
LLM_MODEL      = os.getenv("LLM_MODEL", "gemini-1.5-pro-latest")  # doğru ad

# ====== SINGLETONS ======
_embedder: Optional[SentenceTransformer] = None
_index: Optional[faiss.Index] = None
_chunks: Optional[List[Dict]] = None
_gemini_model: Optional[genai.GenerativeModel] = None


COURSE_MAP = {
    "ceng_102":           ["ceng102", "ceng 102", "programlama", "python dersi", "ceng_102"],
    "computer_networks":  ["computer networks", "bilgisayar ağları", "ağ", "tcp", "udp", "cn", "computer_networks"],
    "plc_204":            ["plc", "ladder", "otomasyon", "plc204", "plc 204", "plc_204"],
}


def _load_embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMB_MODEL_NAME)
    return _embedder

def _load_index():
    global _index
    if _index is None:
        if not os.path.exists(FAISS_PATH):
            raise FileNotFoundError(f"FAISS index not found: {FAISS_PATH}")
        _index = faiss.read_index(FAISS_PATH)
    return _index

def _load_chunks():
    global _chunks
    if _chunks is None:
        if not os.path.exists(CHUNKS_PATH):
            raise FileNotFoundError(f"Chunks file not found: {CHUNKS_PATH}")
        with open(CHUNKS_PATH, "rb") as f:
            _chunks = pickle.load(f)
    return _chunks

def _embed(texts: List[str]) -> np.ndarray:
    emb = _load_embedder()
    vecs = emb.encode(texts, normalize_embeddings=True)
    return np.asarray(vecs, dtype="float32")

def _infer_course(source_name: str) -> Optional[str]:
    """Kaynak adına göre kaba kurs çıkarımı (dosya/klasör adından)."""
    s = (source_name or "").lower()
    for key, kws in COURSE_MAP.items():
        if key in s or any(k in s for k in kws):
            return key
    return None

def _normalize_record(rec) -> Dict:
    """LangChain Document veya dict olabilir; her durumda dict döndür."""
    if hasattr(rec, "page_content"):  
        text = getattr(rec, "page_content", "")
        meta = getattr(rec, "metadata", {}) or {}
        src  = meta.get("source", "unknown.pdf")
        page = meta.get("page")
        course = meta.get("course") or _infer_course(src)
        return {"text": text, "source": src, "page": page, "course": course}
    
    src  = rec.get("source","unknown.pdf")
    page = rec.get("page")
    course = rec.get("course") or _infer_course(src)
    return {"text": rec.get("text",""), "source": src, "page": page, "course": course}

def detect_course(user_text: str) -> Optional[str]:
    t = (user_text or "").lower()
    for key, kws in COURSE_MAP.items():
        if key in t or any(re.search(r"\b"+re.escape(k)+r"\b", t) for k in kws):
            return key
    return None

def _retrieve(query: str, k: int = 5, only_course: Optional[str] = None) -> List[Dict]:
    idx = _load_index()
    raw_chunks = _load_chunks()
    qv = _embed([query])
    D, I = idx.search(qv, max(k*4, k))  
    hits: List[Dict] = []
    for i, score in zip(I[0], D[0]):
        if i == -1:
            continue
        rec = _normalize_record(raw_chunks[i])
        if only_course and rec.get("course") and rec["course"] != only_course:
            continue
        hits.append({
            "text": rec["text"],
            "source": rec["source"],
            "page": rec.get("page"),
            "course": rec.get("course"),
            "score": float(score),
        })
        if len(hits) >= k:
            break

    # filtre sebebiyle k dolmadıysa genel havuzdan tamamlama
    if only_course and len(hits) < max(3, k//2):
        for i, score in zip(I[0], D[0]):
            if i == -1:
                continue
            rec = _normalize_record(raw_chunks[i])
            tag = (rec["source"], rec.get("page"))
            if any((h["source"], h.get("page")) == tag for h in hits):
                continue
            hits.append({
                "text": rec["text"],
                "source": rec["source"],
                "page": rec.get("page"),
                "course": rec.get("course"),
                "score": float(score),
            })
            if len(hits) >= k:
                break
    return hits

def _sources(hits: List[Dict]) -> List[str]:
    out, seen = [], set()
    for h in hits:
        tag = f"{h['source']} s.{h['page']}" if h.get("page") else h['source']
        if tag not in seen:
            out.append(tag); seen.add(tag)
    return out

def _ctx_block(hits: List[Dict]) -> str:
    if not hits:
        return "(bağlam bulunamadı)"
    parts=[]
    for h in hits:
        tag = f"{h['source']} s.{h['page']}" if h.get("page") else h['source']
        parts.append(f"[{tag}]\n{h['text']}")
    return "\n\n---\n\n".join(parts)

# GEMINI 
def _load_llm() -> genai.GenerativeModel:
    """Gemini modelini (doğru adla) bir kez başlat."""
    global _gemini_model
    if _gemini_model is not None:
        return _gemini_model

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY bulunamadı. .env dosyasına ekleyin.")

    genai.configure(api_key=api_key)

    name = LLM_MODEL  # "gemini-1.5-pro-latest" varsayılan
    try:
        _gemini_model = genai.GenerativeModel(name)
    except Exception:
        # model adı desteklenmiyorsa flash-latest'e düş
        _gemini_model = genai.GenerativeModel("gemini-1.5-flash-latest")
    return _gemini_model

SYS = (
   "You are EduRAG. Default language: Turkish. "
  "Cevap verirken SADECE [CONTEXT] bölümündeki bilgilere dayan. "
  "CONTEXT yetersizse uydurma yapma; 'Bağlam yetersiz' de ve "
  "gerekirse kullanıcıdan daha spesifik konu veya belge talep et. "
  "Yanıtın sonunda bağlamdan gelen sayfa/başlıkları kısa parantez içinde göster (örn: [source.pdf s.3])."
)

def _prompt(mode: str, question: str, ctx: str) -> str:
    if mode == "summary":
        req = "Bu bağlamı kısa ve maddeli özetle (5–8 madde)."
    elif mode == "quiz":
        req = "Bağlamdan 3 adet çoktan seçmeli soru üret (A–D seçenekleri, sonunda cevap anahtarı)."
    else:
        req = "Soruyu yanıtla. Bağlam varsa ondan yararlan, bağlam dışında uydurma yapma."
    return f"{SYS}\n\n[USER]\n{question}\n\n[CONTEXT]\n{ctx}\n\n[REQUEST]\n{req}"

def _gemini_call(prompt: str) -> str:
    model = _load_llm()
    # 0.8.x sürümünde string doğrudan verilebilir
    resp = model.generate_content(prompt)
    return getattr(resp, "text", str(resp))

# PUBLIC API 
def reply(user_text: str, k: int = 5) -> Tuple[str, List[str], Optional[str]]:
    """
    Komut algılama:
      - 'özet' / 'özet ver'  -> summary
      - 'soru üret' / 'quiz' -> quiz
      - aksi halde            -> qna
    Ders algılama:
      - ifadedeki anahtar kelimelere göre course seç
    """
    t = user_text.strip()
    low = t.lower()


    if low.startswith("özet") or "özet ver" in low:
        mode = "summary"
        query = re.sub(r"^özet( ver)?[:\- ]*", "", t, flags=re.I).strip() or "özet"
    elif low.startswith("soru üret") or "soru üret" in low or "quiz" in low:
        mode = "quiz"
        query = re.sub(r"^soru üret[:\- ]*", "", t, flags=re.I).strip() or "konu"
    else:
        mode = "qna"
        query = t

    course = detect_course(low)  # None olabilir
    hits = _retrieve(query, k=k, only_course=course)
    ctx = _ctx_block(hits)

    prompt = _prompt(mode, t, ctx)
    ans = _gemini_call(prompt)
    return ans, _sources(hits), course

#  wrapperlar 
def ask(q: str, course: Optional[str] = None):
    if course:
        q = f"{q} ({course})"
    ans, srcs, _ = reply(q)
    return ans, srcs

def summarize(hint: str):
    ans, srcs, _ = reply(f"özet: {hint}")
    return ans, srcs

def quiz(hint: str):
    ans, srcs, _ = reply(f"soru üret: {hint}")
    return ans, srcs
