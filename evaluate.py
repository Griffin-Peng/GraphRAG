import os
import json
import random
import re
from openai import OpenAI
import chromadb
from sentence_transformers import SentenceTransformer
from neo4j import GraphDatabase
from dotenv import load_dotenv

# ===== 配置 =====
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))

DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
NEO4J_URI      = os.getenv("NEO4J_URI", "neo4j://127.0.0.1:7687")
NEO4J_USER     = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")
CHROMA_DIR     = os.path.join(BASE_DIR, os.getenv("CHROMA_DIR", "chroma_db"))
MODEL_NAME     = os.getenv("EMBED_MODEL", "paraphrase-multilingual-MiniLM-L12-v2")
HOP_QA_DIR     = os.path.join(BASE_DIR, os.getenv("HOP_QA_DIR", "hop_qa"))

ONE_HOP_DIR   = os.path.join(HOP_QA_DIR, "one_hop_qa")
TWO_HOP_DIR   = os.path.join(HOP_QA_DIR, "two_hop_qa")
MULTI_HOP_DIR = os.path.join(HOP_QA_DIR, "multi_hop_qa")
RESULT_FILE   = os.path.join(BASE_DIR, "eval_results.json")

MAX_FILES_PER_TYPE     = 50
MAX_QUESTIONS_PER_FILE = 5
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

# ===== 初始化 =====
print("初始化组件...")
llm = OpenAI(api_key=DASHSCOPE_API_KEY,
             base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
embed_model   = SentenceTransformer(MODEL_NAME)
chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
collection    = chroma_client.get_collection("kg_contexts")
neo4j_driver  = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
print("初始化完成！\n")


# ===== 数据加载 =====
def load_questions(folder, max_files, max_per_file):
    files = [f for f in os.listdir(folder) if f.endswith('.json')]
    if len(files) > max_files:
        files = random.sample(files, max_files)
    questions = []
    for fname in files:
        fpath = os.path.join(folder, fname)
        try:
            with open(fpath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            items = data if isinstance(data, list) else [data]
            valid = [q for q in items
                     if all(k in q for k in ['question', 'options', 'correct_answer'])]
            questions.extend(random.sample(valid, min(max_per_file, len(valid))))
        except Exception as e:
            print(f"  跳过 {fname}: {e}")
    return questions


# ===== 上下文构建 =====
def get_rag_context(question, top_k=8, threshold=0.75):
    vec = embed_model.encode([question]).tolist()
    results = collection.query(query_embeddings=vec, n_results=top_k)
    
    lines = []
    for doc, meta, dist in zip(results['documents'][0],
                                results['metadatas'][0],
                                results['distances'][0]):
        similarity = 1 - dist
        if similarity < threshold:  # 低于阈值直接跳过
            continue
        lines.append(
            f"- {meta['subject']} --[{meta['relation']}]--> {meta['object']}: {doc}")
    return "\n".join(lines)


def get_graphrag_context(question, top_k=8, threshold=0.75):
    vec = embed_model.encode([question]).tolist()
    results = collection.query(query_embeddings=vec, n_results=top_k)

    vector_lines = []
    entity_names = []
    for doc, meta, dist in zip(results['documents'][0],
                                results['metadatas'][0],
                                results['distances'][0]):
        similarity = 1 - dist
        if similarity < threshold:
            continue
        vector_lines.append(
            f"- {meta['subject']} --[{meta['relation']}]--> {meta['object']}: {doc}")
        # 只从高置信度结果里取实体做图谱扩展
        entity_names += [meta['subject'], meta['object']]

    entity_names = list(set(entity_names))[:6]

    graph_lines = []
    if entity_names:
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
            for record in session.run(cypher, names=entity_names):
                rel_str = " -> ".join(record["rels"])
                graph_lines.append(
                    f"- {record['start']} --[{rel_str}]--> {record['end']}")

    parts = []
    if vector_lines:
        parts.append("【语义检索】\n" + "\n".join(vector_lines))
    if graph_lines:
        parts.append("【图谱路径】\n" + "\n".join(graph_lines))
    return "\n".join(parts)


# ===== LLM 调用 =====
def build_prompt(question, options, context):
    opts_str = "\n".join(f"{k}: {v}" for k, v in options.items())
    if context:
        return (f"请根据以下参考信息回答问题，只输出选项字母（A/B/C/D），不要解释。\n\n"
                f"参考信息：\n{context}\n\n"
                f"问题：{question}\n选项：\n{opts_str}\n\n答案：")
    else:
        return (f"请回答以下单选题，只输出选项字母（A/B/C/D），不要解释。\n\n"
                f"问题：{question}\n选项：\n{opts_str}\n\n答案：")


def call_llm(prompt):
    try:
        resp = llm.chat.completions.create(
            model="qwen-plus",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=10,
            temperature=0.0
        )
        raw   = resp.choices[0].message.content.strip().upper()
        match = re.search(r'[ABCD]', raw)
        return match.group(0) if match else "X"
    except Exception as e:
        print(f"    LLM调用失败: {e}")
        return "X"


# ===== 评测 =====
def evaluate(questions, label):
    print(f"\n{'='*50}")
    print(f"评测：{label}（共 {len(questions)} 题）")
    print(f"{'='*50}")

    plain_correct = rag_correct = graphrag_correct = 0

    for i, q in enumerate(questions):
        question = q['question']
        options  = q['options']
        correct  = q['correct_answer'].strip().upper()

        plain_ans    = call_llm(build_prompt(question, options, ""))
        rag_ans      = call_llm(build_prompt(question, options, get_rag_context(question)))
        graphrag_ans = call_llm(build_prompt(question, options, get_graphrag_context(question)))

        plain_correct    += (plain_ans == correct)
        rag_correct      += (rag_ans == correct)
        graphrag_correct += (graphrag_ans == correct)

        if (i + 1) % 10 == 0:
            n = i + 1
            print(f"  进度 {n}/{len(questions)} | "
                  f"Plain {plain_correct}/{n} | "
                  f"RAG {rag_correct}/{n} | "
                  f"GraphRAG {graphrag_correct}/{n}")

    n = len(questions)
    summary = {
        "total":        n,
        "plain_acc":    round(plain_correct    / n * 100, 1),
        "rag_acc":      round(rag_correct      / n * 100, 1),
        "graphrag_acc": round(graphrag_correct / n * 100, 1),
    }
    print(f"\n  Plain    {summary['plain_acc']}%")
    print(f"  RAG      {summary['rag_acc']}%")
    print(f"  GraphRAG {summary['graphrag_acc']}%")
    return summary


# ===== 主程序 =====
def main():
    print("加载题目...")
    one_qs   = load_questions(ONE_HOP_DIR,   MAX_FILES_PER_TYPE, MAX_QUESTIONS_PER_FILE)
    two_qs   = load_questions(TWO_HOP_DIR,   MAX_FILES_PER_TYPE, MAX_QUESTIONS_PER_FILE)
    multi_qs = load_questions(MULTI_HOP_DIR, MAX_FILES_PER_TYPE, MAX_QUESTIONS_PER_FILE)
    print(f"题目数：one={len(one_qs)}, two={len(two_qs)}, multi={len(multi_qs)}")

    results = {
        "one_hop":   evaluate(one_qs,   "One-Hop QA"),
        "two_hop":   evaluate(two_qs,   "Two-Hop QA"),
        "multi_hop": evaluate(multi_qs, "Multi-Hop QA"),
    }

    with open(RESULT_FILE, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*50}")
    print("最终汇总")
    print(f"{'='*50}")
    print(f"{'类型':<12} {'题数':>5} {'Plain':>8} {'RAG':>8} {'GraphRAG':>10}")
    print("-" * 50)
    for key, res in results.items():
        print(f"{key:<12} {res['total']:>5} "
              f"{res['plain_acc']:>7}% "
              f"{res['rag_acc']:>7}% "
              f"{res['graphrag_acc']:>9}%")
    print(f"\n结果已保存到 {RESULT_FILE}")

    neo4j_driver.close()


if __name__ == "__main__":
    main()