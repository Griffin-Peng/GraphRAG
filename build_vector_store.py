from rdflib import Graph, URIRef, Literal
from rdflib.namespace import RDF
import os
import re
import chromadb
from sentence_transformers import SentenceTransformer
import logging
from dotenv import load_dotenv

# ===== 配置 =====
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))

TTL_DIR    = os.path.join(BASE_DIR, os.getenv("KG_TTL_DIR", "KG_TTL"))
CHROMA_DIR = os.path.join(BASE_DIR, os.getenv("CHROMA_DIR", "chroma_db"))
MODEL_NAME = os.getenv("EMBED_MODEL", "paraphrase-multilingual-MiniLM-L12-v2")

logging.basicConfig(
    filename=os.path.join(BASE_DIR, "vector_log.txt"),
    filemode="w",
    encoding="utf-8",
    level=logging.INFO,
    format="%(message)s"
)

def log(msg):
    print(msg)
    logging.info(msg)

# ===== 解析函数（复用之前的逻辑）=====
def extract_ttl_content(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    blocks = re.findall(r'```turtle\s*(.*?)```', content, re.DOTALL)
    if not blocks:
        return None
    prefix = (
        "@prefix : <http://example.org/> .\n"
        "@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .\n"
        "@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .\n"
        "@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .\n"
        "@prefix owl: <http://www.w3.org/2002/07/owl#> .\n\n"
    )
    return prefix + '\n'.join(blocks)

def extract_contexts(filepath):
    """提取文件中所有三元组及其contextText"""
    ttl_content = extract_ttl_content(filepath)
    if ttl_content is None:
        return []

    chunks = re.split(r'\n\s*\n', ttl_content)
    prefix = (
        "@prefix : <http://example.org/> .\n"
        "@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .\n"
        "@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .\n"
        "@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .\n"
        "@prefix owl: <http://www.w3.org/2002/07/owl#> .\n\n"
    )

    results = []
    for chunk in chunks:
        chunk = chunk.strip()
        if not chunk or len(re.findall(r':[A-Z]', chunk)) < 1:
            continue
        try:
            g = Graph()
            g.parse(data=prefix + chunk, format="turtle")

            # 从这个块里提取主体、关系、客体、contextText
            subject = relation = obj = context = section = None
            for s, p, o in g:
                p_name = str(p).split("/")[-1]
                if p_name == "contextText" and isinstance(o, Literal):
                    context = str(o)
                elif p_name == "sourceSection" and isinstance(o, Literal):
                    section = str(o)
                elif isinstance(o, URIRef) and p != RDF.type and p_name not in {'sourceChunk', 'sourceSection'}:
                    subject = str(s).split("/")[-1]
                    relation = p_name
                    obj = str(o).split("/")[-1]

            # 只收录有完整信息的条目
            if subject and relation and obj and context:
                results.append({
                    "subject": subject,
                    "relation": relation,
                    "object": obj,
                    "context": context,
                    "section": section or "",
                    "source": os.path.basename(filepath)
                })
        except Exception:
            continue

    return results

def main():
    # 加载向量模型
    log(f"加载向量模型 {MODEL_NAME}...")
    log("首次运行会下载模型文件（约420MB），请耐心等待...")
    model = SentenceTransformer(MODEL_NAME)
    log("模型加载完成！\n")

    # 初始化 ChromaDB
    log(f"初始化向量数据库，存储路径：{CHROMA_DIR}")
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    # 如果集合已存在则删除重建
    try:
        client.delete_collection("kg_contexts")
    except Exception:
        pass
    collection = client.create_collection(
        name="kg_contexts",
        metadata={"hnsw:space": "cosine"}  # 用余弦相似度
    )
    log("向量数据库初始化完成！\n")

    # 遍历所有TTL文件
    files = [f for f in os.listdir(TTL_DIR) if f.endswith('.ttl')]
    log(f"开始处理 {len(files)} 个TTL文件...\n")

    all_records = []
    for i, filename in enumerate(files):
        filepath = os.path.join(TTL_DIR, filename)
        records = extract_contexts(filepath)
        all_records.extend(records)
        if (i + 1) % 20 == 0:
            log(f"已处理 {i+1}/{len(files)} 个文件，累计 {len(all_records)} 条记录")

    log(f"\n共提取 {len(all_records)} 条带上下文的三元组，开始向量化...\n")

    # 批量向量化并存入ChromaDB
    batch_size = 256
    for i in range(0, len(all_records), batch_size):
        batch = all_records[i:i + batch_size]

        texts = [r["context"] for r in batch]
        embeddings = model.encode(texts, show_progress_bar=False).tolist()

        collection.add(
            ids=[f"doc_{i+j}" for j in range(len(batch))],
            embeddings=embeddings,
            documents=texts,
            metadatas=[{
                "subject": r["subject"],
                "relation": r["relation"],
                "object": r["object"],
                "section": r["section"],
                "source": r["source"]
            } for r in batch]
        )

        log(f"向量化进度：{min(i+batch_size, len(all_records))}/{len(all_records)}")

    log(f"\n===== 向量库构建完成 =====")
    log(f"总记录数：{len(all_records)} 条")
    log(f"存储路径：{CHROMA_DIR}")

    # 简单测试一下检索
    log("\n--- 测试检索 ---")
    test_query = "federated learning continual learning"
    query_vec = model.encode([test_query]).tolist()
    results = collection.query(query_embeddings=query_vec, n_results=3)
    log(f"查询：'{test_query}'")
    log("Top3 结果：")
    for j, (doc, meta) in enumerate(zip(results['documents'][0], results['metadatas'][0])):
        log(f"  [{j+1}] {meta['subject']} --[{meta['relation']}]--> {meta['object']}")
        log(f"       {doc[:80]}...")

if __name__ == "__main__":
    main()
