from neo4j import GraphDatabase
import chromadb
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import os
from dotenv import load_dotenv

# ===== 配置 =====
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))

NEO4J_URI      = os.getenv("NEO4J_URI", "neo4j://127.0.0.1:7687")
NEO4J_USER     = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")
CHROMA_DIR     = os.path.join(BASE_DIR, os.getenv("CHROMA_DIR", "chroma_db"))
MODEL_NAME     = os.getenv("EMBED_MODEL", "paraphrase-multilingual-MiniLM-L12-v2")
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")

# ===== 初始化所有组件 =====
print("初始化系统组件...")

# 1. 向量模型
print("  加载向量模型...")
embed_model = SentenceTransformer(MODEL_NAME)

# 2. ChromaDB
print("  连接向量数据库...")
chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
collection = chroma_client.get_collection("kg_contexts")

# 3. Neo4j
print("  连接图数据库...")
neo4j_driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

# 4. 通义千问 LLM（兼容 OpenAI 接口）
print("  连接通义千问...")
llm_client = OpenAI(
    api_key=DASHSCOPE_API_KEY,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

print("所有组件初始化完成！\n")


# ===== 检索函数 =====

def vector_search(query, top_k=5):
    """向量检索：找语义相近的contextText"""
    query_vec = embed_model.encode([query]).tolist()
    results = collection.query(query_embeddings=query_vec, n_results=top_k)

    hits = []
    for doc, meta in zip(results['documents'][0], results['metadatas'][0]):
        hits.append({
            "context": doc,
            "subject": meta["subject"],
            "relation": meta["relation"],
            "object": meta["object"],
            "source": meta["source"]
        })
    return hits


def graph_search(entity_names, depth=2):
    """图谱检索：从实体出发，找相关关系路径"""
    if not entity_names:
        return []

    with neo4j_driver.session(database="neo4j") as session:
        query = """
        UNWIND $names AS name
        MATCH (e:Entity)
        WHERE e.name CONTAINS name OR name CONTAINS e.name
        WITH e LIMIT 5
        MATCH path = (e)-[r:RELATION*1..2]->(target:Entity)
        RETURN e.name AS start,
               [rel in relationships(path) | rel.type] AS relations,
               target.name AS end,
               [rel in relationships(path) | rel.source] AS sources
        LIMIT 20
        """
        result = session.run(query, names=entity_names)
        paths = []
        for record in result:
            paths.append({
                "start": record["start"],
                "relations": record["relations"],
                "end": record["end"],
                "source": record["sources"][0] if record["sources"] else ""
            })
    return paths


def extract_key_entities(query):
    """从查询中提取可能的实体关键词（简单启发式）"""
    # 过滤常见停用词，保留可能是实体的词
    stopwords = {"what", "how", "why", "when", "where", "which", "who",
                 "is", "are", "the", "a", "an", "of", "in", "to", "for",
                 "and", "or", "does", "do", "can", "will", "has", "have"}
    words = query.replace("?", "").replace(",", "").split()
    entities = [w for w in words if w.lower() not in stopwords and len(w) > 3]
    return entities[:5]  # 最多取5个关键词


def build_context(vector_hits, graph_paths):
    """把两路检索结果拼成给LLM的上下文"""
    context_parts = []

    if vector_hits:
        context_parts.append("=== 相关文献上下文 ===")
        for i, hit in enumerate(vector_hits):
            context_parts.append(
                f"[{i+1}] {hit['subject']} --[{hit['relation']}]--> {hit['object']}\n"
                f"    {hit['context']}"
            )

    if graph_paths:
        context_parts.append("\n=== 知识图谱关系路径 ===")
        for path in graph_paths[:10]:  # 最多10条路径
            relations_str = " -> ".join(path["relations"])
            context_parts.append(
                f"  {path['start']} --[{relations_str}]--> {path['end']}"
            )

    return "\n".join(context_parts)


def ask(question, verbose=False):
    """主问答函数"""
    print(f"\n问题：{question}")
    print("-" * 60)

    # 1. 向量检索
    vector_hits = vector_search(question, top_k=5)
    if verbose:
        print(f"向量检索命中 {len(vector_hits)} 条")

    # 2. 图谱检索
    key_entities = extract_key_entities(question)
    graph_paths = graph_search(key_entities, depth=2)
    if verbose:
        print(f"图谱检索命中 {len(graph_paths)} 条路径")
        print(f"提取实体关键词：{key_entities}")

    # 3. 拼上下文
    context = build_context(vector_hits, graph_paths)
    if verbose:
        print(f"\n--- 上下文 ---\n{context}\n")

    # 4. 调用LLM
    system_prompt = """你是一个专业的AI与计算机科学领域问答助手。
请基于提供的知识图谱上下文回答问题。
回答要准确、简洁，如果上下文中没有足够信息，请如实说明。
请用中文回答。"""

    user_prompt = f"""请基于以下知识图谱上下文回答问题：

{context}

问题：{question}

请给出准确、有条理的回答。"""

    response = llm_client.chat.completions.create(
        model="qwen-plus",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        max_tokens=1000,
        temperature=0.3
    )

    answer = response.choices[0].message.content
    print(f"回答：\n{answer}")
    return answer


# ===== 交互式问答循环 =====
if __name__ == "__main__":
    print("=" * 60)
    print("GraphRAG 问答系统已就绪")
    print("输入问题开始问答，输入 'quit' 退出，输入 'verbose' 切换详细模式")
    print("=" * 60)

    verbose_mode = False

    # 先跑几个测试问题
    test_questions = [
        "What is federated learning and how is it applied?",
        "How does continual learning handle catastrophic forgetting?",
    ]

    print("\n--- 自动测试 ---")
    for q in test_questions:
        ask(q, verbose=False)

    print("\n--- 进入交互模式 ---")
    while True:
        user_input = input("\n请输入问题：").strip()
        if not user_input:
            continue
        if user_input.lower() == 'quit':
            print("退出系统")
            neo4j_driver.close()
            break
        if user_input.lower() == 'verbose':
            verbose_mode = not verbose_mode
            print(f"详细模式：{'开启' if verbose_mode else '关闭'}")
            continue
        ask(user_input, verbose=verbose_mode)