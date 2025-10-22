import os
import streamlit as st
from dotenv import load_dotenv
load_dotenv()

from rag_pipeline import ask  

st.set_page_config(page_title="EduRAG", page_icon="🧠", layout="wide")
st.title("🧠 EduRAG — Ders Notları Asistanı")
st.caption("Gemini + bge-m3 + FAISS + LangChain + Streamlit")

def _normalize_doc(d):
    """
    Her tipten girdiyi (Document, (Document,score), dict, str, tuple, list)
    güvenli biçimde (text, metadata: dict) olarak döndür.
    """
    if isinstance(d, (tuple, list)) and len(d) >= 1 and not hasattr(d, "page_content"):
        d = d[0]

    # LangChain Document
    if hasattr(d, "page_content"):
        text = getattr(d, "page_content", "")
        md = getattr(d, "metadata", {}) or {}
        if not isinstance(md, dict):
            md = {"source": str(md)}
        return text, md

    
    if isinstance(d, dict):
        text = d.get("text") or d.get("content") or d.get("page_content") or ""
        md = d.get("metadata") or d.get("meta") or {}
        if not isinstance(md, dict):
            md = {"source": str(md)}
        return text, md

    
    return str(d), {"source": str(d)}

def render_sources_any(sources):
    """
    sources şu türlerden biri olabilir:
    - [(Document, score), ...]  - [Document, ...]  - [dict, ...]
    - düz string (LLM'in ürettiği 'Kaynaklar:' paragrafı)
    - None
    Bu fonksiyon her durumda güvenli bir markdown üretir.
    """
    
    if not sources:
        return "_(seçili kaynak yok)_"

    
    if isinstance(sources, str):
        return sources if sources.strip() else "_(kaynak metni yok)_"

    
    if isinstance(sources, (list, tuple)):
        lines = []
        for i, h in enumerate(sources, 1):
            _text, md = _normalize_doc(h)
            src   = md.get("source") or md.get("path") or md.get("file") or "Bilinmiyor"
            page  = md.get("page") or md.get("page_number")
            sect  = md.get("heading") or md.get("section") or md.get("unit")
            tag_page = f", s.{page}" if page not in (None, "", "?") else ""
            tag_sect = f" — {sect}" if sect else ""
            course = md.get("course")
            tag_course = f" ({course})" if course else ""
            lines.append(f"- [{src}{tag_page}{tag_sect}]{tag_course}")
        return "\n".join(lines) if lines else "_(seçili kaynak yok)_"

    
    return str(sources)

# İpucu kartı
with st.expander("ℹ️ Nasıl çalışır?", expanded=False):
    st.markdown("""
- Cevaplar **yalnızca** PDF/TXT/MD notlarınızdan gelir.
- Bağlam zayıfsa “Bağlam yetersiz” denir (uydurma yok).
- Aşağıdan dersi seç, sorunuzu yaz, **Çalıştır**.
""")


tab1, tab2, tab3 = st.tabs(["CENG102", "Computer Networks", "PLC 204"])

def qa_tab(label, course_key):
    with label:
        q = st.text_input("Soru", key=f"q_{course_key}", placeholder="Örn: Cache bellek neden kullanılır?")
        col1, col2 = st.columns([1,5])
        go = col1.button("Çalıştır", key=f"go_{course_key}")
        if go and q.strip():
            with st.spinner("Yanıt üretiliyor..."):
                try:
                    ans, srcs = ask(q.strip(), course=course_key)  
                    st.markdown("### Yanıt")
                    st.write(ans)

                    st.markdown("### Kaynaklar")
                    st.markdown(render_sources_any(srcs))
                except Exception as e:
                    st.error(f"Hata: {e}")

qa_tab(tab1, "ceng_102")
qa_tab(tab2, "computer_networks")
qa_tab(tab3, "plc_204")
