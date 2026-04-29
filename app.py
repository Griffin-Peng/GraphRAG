import streamlit as st
from neo4j import GraphDatabase
import chromadb
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import os
import time
from dotenv import load_dotenv

# ==================== 配置 ====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))

DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
NEO4J_URI      = os.getenv("NEO4J_URI", "neo4j://127.0.0.1:7687")
NEO4J_USER     = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")
MODEL_NAME     = os.getenv("EMBED_MODEL", "paraphrase-multilingual-MiniLM-L12-v2")
CHROMA_DIR     = os.path.join(BASE_DIR, os.getenv("CHROMA_DIR", "chroma_db"))

EXAMPLES = [
    "What is federated learning?",
    "How does continual learning handle catastrophic forgetting?",
    "What are the advantages of graph neural networks?",
]

# ==================== 页面配置 ====================
st.set_page_config(
    page_title="GraphRAG 知识问答",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== 样式 ====================
def get_styles(dark_mode):
    if dark_mode:
        bg       = "#0d0f12"
        sidebar  = "#111318"
        border   = "#1e2530"
        text     = "#e2e8f0"
        subtext  = "#64748b"
        accent   = "#38bdf8"
        msguser  = "#1e2530"
        msgbot   = "#111827"
        botborder= "#1e2d3d"
        evidence = "#0a0c10"
        badge    = "#1e2530"
        input_bg = "#111318"
        triple_border = "#1a1f28"
        entity   = "#7dd3fc"
        relation = "#86efac"
    else:
        bg       = "#f8fafc"
        sidebar  = "#f1f5f9"
        border   = "#e2e8f0"
        text     = "#0f172a"
        subtext  = "#94a3b8"
        accent   = "#0284c7"
        msguser  = "#e0f2fe"
        msgbot   = "#ffffff"
        botborder= "#bae6fd"
        evidence = "#f8fafc"
        badge    = "#e2e8f0"
        input_bg = "#ffffff"
        triple_border = "#e2e8f0"
        entity   = "#0369a1"
        relation = "#15803d"

    return f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');
    * {{ font-family: 'IBM Plex Sans', sans-serif; }}
    #MainMenu, footer {{ visibility: hidden; }}
    /* header 保留，因为左上角包含侧边栏展开/收起按钮 */

    .stApp {{ background: {bg}; color: {text}; }}
    section[data-testid="stSidebar"] {{ background: {sidebar} !important; border-right: 1px solid {border}; }}

    .hero {{
        padding: 1.2rem 0 0.8rem 0;
        border-bottom: 1px solid {border};
        margin-bottom: 1rem;
    }}
    .hero h1 {{
        font-family: 'IBM Plex Mono', monospace;
        font-size: 1.5rem;
        font-weight: 600;
        color: {accent};
        margin: 0;
    }}
    .hero p {{
        color: {subtext};
        font-size: 0.82rem;
        margin: 0.3rem 0 0 0;
        font-family: 'IBM Plex Mono', monospace;
    }}

    /* 聊天消息微调 */
    div[data-testid="stChatMessage"] {{
        border-radius: 12px;
        margin-bottom: 0.6rem;
        padding: 0.5rem 0.75rem;
    }}
    div[data-testid="stChatMessageContent"] {{
        font-size: 0.95rem;
        line-height: 1.7;
    }}

    .badge {{
        display: inline-block;
        background: {badge};
        border: 1px solid {border};
        border-radius: 4px;
        padding: 0.18rem 0.55rem;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.7rem;
        color: {subtext};
        margin-right: 0.4rem;
        margin-bottom: 0.4rem;
    }}
    .badge-blue {{ border-color: {accent}33; color: {accent}; }}
    .badge-green {{ border-color: {relation}33; color: {relation}; }}

    .sidebar-section {{
        background: {bg};
        border: 1px solid {border};
        border-radius: 6px;
        padding: 0.75rem;
        margin-bottom: 0.75rem;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.73rem;
    }}
    .sidebar-label {{
        color: {subtext};
        font-size: 0.62rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-bottom: 0.4rem;
        font-weight: 600;
    }}

    hr {{ border-color: {border} !important; }}

    .stButton > button {{
        background: {bg} !important;
        border: 1px solid {accent} !important;
        color: {accent} !important;
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: 0.78rem !important;
        border-radius: 8px !important;
        padding: 0.5rem 1.2rem !important;
        transition: all 0.15s !important;
    }}
    .stButton > button:hover {{
        background: {accent}18 !important;
    }}
    div[data-testid="stSelectbox"] > div > div {{
        background: {input_bg} !important;
        border-color: {border} !important;
        color: {text} !important;
    }}
    .stSlider > div > div > div > div {{
        background: {accent} !important;
    }}
    </style>
    """

# ==================== Session State ====================
defaults = {
    "messages": [],
    "dark_mode": True,
    "mode": "GraphRAG",
    "threshold": 0.75,
    "show_evidence": True,
    "examples_shown": True,
    "backend_errors": [],
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

st.markdown(get_styles(st.session_state.dark_mode), unsafe_allow_html=True)

# ==================== 初始化组件 ====================
@st.cache_resource(show_spinner=False)
def init_components():
    status = {
        "embed": None, "collection": None,
        "driver": None, "llm": None,
        "errors": []
    }
    try:
        status["embed"] = SentenceTransformer(MODEL_NAME, local_files_only=True)
    except Exception as e:
        status["errors"].append(f"Embedding model 加载失败: {e}")
    try:
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        status["collection"] = client.get_collection("kg_contexts")
    except Exception as e:
        status["errors"].append(f"ChromaDB 连接失败: {e}")
    try:
        status["driver"] = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        status["driver"].verify_connectivity()
    except Exception as e:
        status["errors"].append(f"Neo4j 连接失败: {e}")
        status["driver"] = None
    try:
        status["llm"] = OpenAI(
            api_key=DASHSCOPE_API_KEY,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
    except Exception as e:
        status["errors"].append(f"LLM Client 初始化失败: {e}")
    return status

components = init_components()
st.session_state.backend_errors = components["errors"]

embed_model = components["embed"]
collection  = components["collection"]
neo4j_driver= components["driver"]
llm_client  = components["llm"]

backend_ready = all([embed_model, collection, llm_client])

# ==================== 检索函数 ====================
def vector_search(question, top_k=8, threshold=0.75):
    if not embed_model or not collection:
        return []
    vec = embed_model.encode([question]).tolist()
    results = collection.query(query_embeddings=vec, n_results=top_k)
    hits = []
    for doc, meta, dist in zip(results['documents'][0],
                                results['metadatas'][0],
                                results['distances'][0]):
        if 1 - dist >= threshold:
            hits.append({"context": doc, **meta, "score": round(1-dist, 3)})
    return hits

def graph_search(entity_names):
    if not neo4j_driver or not entity_names:
        return []
    with neo4j_driver.session(database="neo4j") as session:
        cypher = """
        UNWIND $names AS name
        MATCH (e:Entity {name: name})
        WITH e LIMIT 6
        MATCH path = (e)-[r:RELATION*1..2]->(target:Entity)
        RETURN e.name AS start,
               [rel in relationships(path) | rel.type] AS rels,
               target.name AS end
        LIMIT 15
        """
        return [{"start": r["start"], "rels": r["rels"], "end": r["end"]}
                for r in session.run(cypher, names=entity_names)]

def ask_stream(question, mode, threshold):
    vector_hits, graph_paths = [], []
    if mode in ["RAG", "GraphRAG"]:
        vector_hits = vector_search(question, threshold=threshold)
    if mode == "GraphRAG" and vector_hits:
        names = []
        for h in vector_hits:
            names.extend([h.get("subject", ""), h.get("object", "")])
        names = list(set([n for n in names if n]))[:6]
        graph_paths = graph_search(names)

    context = ""
    if vector_hits:
        lines = [f"- {h['subject']} --[{h['relation']}]--> {h['object']}: {h['context']}"
                 for h in vector_hits]
        context += "【语义检索】\n" + "\n".join(lines)
    if graph_paths:
        lines = [f"- {p['start']} --[{' -> '.join(p['rels'])}]--> {p['end']}"
                 for p in graph_paths]
        context += "\n\n【图谱路径】\n" + "\n".join(lines)

    prompt = (f"Based on the following context, answer the question in Chinese. Be accurate and concise.\n\nContext:\n{context}\n\nQuestion: {question}"
              if context else f"Please answer in Chinese: {question}")

    if not llm_client:
        raise RuntimeError("LLM 客户端未初始化")

    stream = llm_client.chat.completions.create(
        model="qwen-plus",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=800,
        temperature=0.3,
        stream=True,
    )
    return stream, vector_hits, graph_paths

# ==================== 证据渲染 ====================
def render_evidence(msg):
    vh = msg.get("vector_hits", [])
    gp = msg.get("graph_paths", [])
    if not vh and not gp:
        return

    dark = st.session_state.dark_mode
    accent = "#38bdf8" if dark else "#0284c7"
    entity_color = "#7dd3fc" if dark else "#0369a1"
    relation_color = "#86efac" if dark else "#15803d"
    subtext = "#64748b" if dark else "#94a3b8"
    triple_border_color = "#1a1f28" if dark else "#e2e8f0"

    label = f"📎 检索证据（{len(vh)} 条语义命中"
    if gp:
        label += f"，{len(gp)} 条图谱路径"
    label += "）"

    with st.expander(label, expanded=False):
        if vh:
            st.markdown(
                f"<div style='font-family:IBM Plex Mono,monospace;font-size:0.65rem;color:{accent};"
                f"text-transform:uppercase;letter-spacing:0.1em;margin-bottom:0.5rem;font-weight:600'>"
                f"语义检索结果</div>",
                unsafe_allow_html=True
            )
            for h in vh:
                ctx = h.get('context', '')[:120]
                st.markdown(
                    f"<div style='font-family:IBM Plex Mono,monospace;font-size:0.75rem;color:{subtext};"
                    f"padding:0.35rem 0;border-bottom:1px solid {triple_border_color};line-height:1.6'>"
                    f"<span style='color:{entity_color}'>{h.get('subject','')}</span> "
                    f"<span style='color:{relation_color}'>─[{h.get('relation','')}]→</span> "
                    f"<span style='color:{entity_color}'>{h.get('object','')}</span> "
                    f"<span style='opacity:0.5'>({h.get('score',0)})</span><br>"
                    f"<span style='opacity:0.6;font-size:0.68rem'>{ctx}...</span>"
                    f"</div>",
                    unsafe_allow_html=True
                )
        if gp:
            st.markdown(
                f"<div style='font-family:IBM Plex Mono,monospace;font-size:0.65rem;color:{accent};"
                f"text-transform:uppercase;letter-spacing:0.1em;margin:0.8rem 0 0.5rem 0;font-weight:600'>"
                f"图谱路径扩展</div>",
                unsafe_allow_html=True
            )
            for p in gp[:10]:
                rel_str = " → ".join(p.get("rels", []))
                st.markdown(
                    f"<div style='font-family:IBM Plex Mono,monospace;font-size:0.75rem;color:{subtext};"
                    f"padding:0.35rem 0;border-bottom:1px solid {triple_border_color};line-height:1.6'>"
                    f"<span style='color:{entity_color}'>{p.get('start','')}</span> "
                    f"<span style='color:{relation_color}'>─[{rel_str}]→</span> "
                    f"<span style='color:{entity_color}'>{p.get('end','')}</span>"
                    f"</div>",
                    unsafe_allow_html=True
                )

# ==================== 侧边栏 ====================
with st.sidebar:
    st.markdown(f"""
    <div style='font-family:IBM Plex Mono,monospace;font-size:0.65rem;
                color:{"#475569" if st.session_state.dark_mode else "#94a3b8"};
                text-transform:uppercase;letter-spacing:0.1em;
                padding:0.5rem 0 0.8rem 0;
                border-bottom:1px solid {"#1e2530" if st.session_state.dark_mode else "#e2e8f0"};
                margin-bottom:0.8rem'>
        系统配置
    </div>""", unsafe_allow_html=True)

    col_l, col_r = st.columns([2, 1])
    with col_l:
        st.markdown("<div style='font-size:0.82rem;padding-top:0.4rem'>🌓 深色模式</div>",
                    unsafe_allow_html=True)
    with col_r:
        new_dark = st.toggle("", value=st.session_state.dark_mode, key="theme_toggle")
    if new_dark != st.session_state.dark_mode:
        st.session_state.dark_mode = new_dark
        st.rerun()

    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

    st.session_state.mode = st.selectbox(
        "检索模式",
        ["GraphRAG", "RAG", "Plain"],
        index=["GraphRAG", "RAG", "Plain"].index(st.session_state.mode),
        help="GraphRAG=图谱+向量，RAG=仅向量，Plain=直接问LLM"
    )
    st.session_state.threshold = st.slider(
        "相似度阈值", 0.5, 0.95, st.session_state.threshold, 0.05,
        help="低于此分数的检索结果将被过滤"
    )
    st.session_state.show_evidence = st.toggle(
        "显示检索证据", value=st.session_state.show_evidence
    )

    st.markdown("<hr>", unsafe_allow_html=True)

    accent = "#38bdf8" if st.session_state.dark_mode else "#0284c7"
    green  = "#86efac" if st.session_state.dark_mode else "#15803d"
    red    = "#ef4444"

    st.markdown(f"""
    <div class='sidebar-section'>
        <div class='sidebar-label'>服务状态</div>
        <div style='line-height:1.8'>
            {"🟢" if embed_model else "🔴"} Embedding 模型<br>
            {"🟢" if collection else "🔴"} ChromaDB<br>
            {"🟢" if neo4j_driver else "⚪"} Neo4j {"(可选)" if not neo4j_driver else ""}<br>
            {"🟢" if llm_client else "🔴"} LLM API
        </div>
    </div>
    <div class='sidebar-section'>
        <div class='sidebar-label'>知识库</div>
        <span class='badge badge-blue'>17,741 节点</span>
        <span class='badge badge-green'>21,534 向量</span>
    </div>
    <div class='sidebar-section'>
        <div class='sidebar-label'>准确率对比</div>
        <div style='line-height:2;font-size:0.72rem'>
            Plain &nbsp;&nbsp; 94.0 / 90.8 / 89.2<br>
            RAG &nbsp;&nbsp;&nbsp; 95.6 / 93.6 / 89.6<br>
            <span style='color:{accent}'>GraphRAG 96.0 / 94.0 / 90.0</span>
        </div>
        <div style='color:{"#475569" if st.session_state.dark_mode else "#94a3b8"};font-size:0.62rem;margin-top:0.3rem'>
            1-hop / 2-hop / multi-hop
        </div>
    </div>
    """, unsafe_allow_html=True)

    if st.button("🗑 清空对话"):
        st.session_state.messages = []
        st.session_state.examples_shown = True
        st.rerun()

    if st.session_state.backend_errors:
        st.markdown(f"<div style='color:{red};font-size:0.75rem;margin-top:1rem'>"
                    f"⚠️ 部分服务异常，请检查配置<br>"
                    f"{'<br>'.join(st.session_state.backend_errors)}</div>",
                    unsafe_allow_html=True)

# ==================== 主区域 ====================
st.markdown("""
<div class='hero'>
    <h1>⬡ GraphRAG 知识问答</h1>
    <p>// 知识图谱增强检索 · 200篇AI/CS论文 · Neo4j + ChromaDB + Qwen</p>
</div>
""", unsafe_allow_html=True)

# 示例问题
if st.session_state.examples_shown and not st.session_state.messages:
    st.markdown("<div style='text-align:center;opacity:0.5;font-size:0.8rem;margin-bottom:0.8rem'>试试这些问题</div>",
                unsafe_allow_html=True)
    cols = st.columns(3)
    for i, (col, ex) in enumerate(zip(cols, EXAMPLES)):
        with col:
            if st.button(ex, key=f"ex_{i}", use_container_width=True):
                st.session_state.messages.append({
                    "role": "user", "content": ex,
                    "mode": st.session_state.mode
                })
                st.session_state.examples_shown = False
                st.rerun()

# 聊天记录
for msg in st.session_state.messages:
    avatar = "🧑‍💻" if msg["role"] == "user" else "🤖"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and st.session_state.show_evidence:
            render_evidence(msg)

# Chat input
if not backend_ready:
    st.chat_input("⛔ 后端服务未就绪，请检查配置", disabled=True)
else:
    if prompt := st.chat_input("输入问题后按 Enter 发送..."):
        st.session_state.messages.append({
            "role": "user", "content": prompt,
            "mode": st.session_state.mode
        })
        st.session_state.examples_shown = False
        st.rerun()

# 处理待回复
if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
    last_msg = st.session_state.messages[-1]
    question = last_msg["content"]
    mode = last_msg.get("mode", st.session_state.mode)

    with st.chat_message("assistant", avatar="🤖"):
        status_area = st.empty()
        response_area = st.empty()

        try:
            start_time = time.time()

            status_area.markdown("🔍 **检索中...**")
            stream, vh, gp = ask_stream(question, mode, st.session_state.threshold)

            status_text = f"✅ 检索完成"
            if vh:
                status_text += f" · {len(vh)} 条语义"
            if gp:
                status_text += f" · {len(gp)} 条图谱"
            status_text += f" · 生成中..."
            status_area.markdown(f"<span style='font-size:0.8rem;color:#64748b'>{status_text}</span>",
                               unsafe_allow_html=True)

            full_response = ""
            for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    full_response += delta
                    response_area.markdown(full_response + "▌")

            response_area.markdown(full_response)
            status_area.empty()

            elapsed = round(time.time() - start_time, 2)
            st.caption(f"⏱ {elapsed}s · {mode} 模式")

            st.session_state.messages.append({
                "role": "assistant",
                "content": full_response,
                "mode": mode,
                "vector_hits": vh,
                "graph_paths": gp,
                "elapsed": elapsed,
            })
        except Exception as e:
            response_area.error(f"生成失败：{e}")
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"抱歉，处理问题时出错：{e}",
                "mode": mode,
                "vector_hits": [], "graph_paths": [],
            })

    st.rerun()
