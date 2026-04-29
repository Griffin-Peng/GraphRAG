from rdflib import Graph, URIRef
from rdflib.namespace import RDF
from neo4j import GraphDatabase
import os
import re
import logging
from dotenv import load_dotenv

# ===== 配置 =====
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))

NEO4J_URI      = os.getenv("NEO4J_URI", "neo4j://127.0.0.1:7687")
NEO4J_USER     = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")
TTL_DIR        = os.path.join(BASE_DIR, os.getenv("KG_TTL_DIR", "KG_TTL"))

# ===== 日志设置 =====
logging.basicConfig(
    filename=os.path.join(BASE_DIR, "import_log.txt"),
    filemode="w",
    encoding="utf-8",
    level=logging.INFO,
    format="%(message)s"
)

def log(msg):
    print(msg)
    logging.info(msg)

# ===== 解析函数 =====
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

def parse_triples(filepath):
    ttl_content = extract_ttl_content(filepath)
    if ttl_content is None:
        log(f"  [跳过] 未找到turtle块: {os.path.basename(filepath)}")
        return []

    # 按段落分割，逐段尝试解析，跳过有问题的段落
    prefix = (
        "@prefix : <http://example.org/> .\n"
        "@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .\n"
        "@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .\n"
        "@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .\n"
        "@prefix owl: <http://www.w3.org/2002/07/owl#> .\n\n"
    )

    # 按空行分割成独立的三元组块
    chunks = re.split(r'\n\s*\n', ttl_content)

    skip_props = {'sourceChunk', 'sourceSection', 'contextText'}
    triples = []

    for chunk in chunks:
        chunk = chunk.strip()
        if not chunk:
            continue
        # 只处理实体关系行（跳过纯属性块）
        # 快速预判：如果chunk里没有两个以上的:大写实体，跳过
        if len(re.findall(r':[A-Z]', chunk)) < 2:
            continue
        try:
            g = Graph()
            g.parse(data=prefix + chunk, format="turtle")
            for s, p, o in g:
                if isinstance(o, URIRef) and p != RDF.type:
                    p_name = str(p).split("/")[-1]
                    if p_name in skip_props:
                        continue
                    triples.append({
                        "subject": str(s).split("/")[-1],
                        "relation": p_name,
                        "object": str(o).split("/")[-1]
                    })
        except Exception:
            continue  # 这个块有问题就跳过，不影响其他块

    return triples

# ===== 导入Neo4j =====
def import_triples(driver, triples, source_file):
    with driver.session(database="neo4j") as session:
        query = """
        UNWIND $triples AS triple
        MERGE (s:Entity {name: triple.subject})
        MERGE (o:Entity {name: triple.object})
        MERGE (s)-[r:RELATION {type: triple.relation}]->(o)
        ON CREATE SET r.source = triple.source
        """
        data = [{"subject": t["subject"],
                 "relation": t["relation"],
                 "object": t["object"],
                 "source": source_file} for t in triples]
        session.run(query, triples=data)

def main():
    log("连接 Neo4j...")
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    try:
        driver.verify_connectivity()
        log("连接成功！\n")
    except Exception as e:
        log(f"连接失败：{e}")
        log("请确认 Neo4j 实例已启动（RUNNING 状态）")
        return

    files = [f for f in os.listdir(TTL_DIR) if f.endswith('.ttl')]
    log(f"共找到 {len(files)} 个TTL文件，开始导入...\n")

    total_triples = 0
    failed_files = []

    for i, filename in enumerate(files):
        filepath = os.path.join(TTL_DIR, filename)
        triples = parse_triples(filepath)

        if not triples:
            failed_files.append(filename)
            continue

        import_triples(driver, triples, filename)
        total_triples += len(triples)
        log(f"[{i+1}/{len(files)}] {filename} → {len(triples)} 条三元组")

    driver.close()

    log(f"\n===== 导入完成 =====")
    log(f"成功处理：{len(files) - len(failed_files)} 个文件")
    log(f"导入三元组：{total_triples} 条")
    if failed_files:
        log(f"失败文件：{len(failed_files)} 个")
        for f in failed_files:
            log(f"  - {f}")
    log("\n日志已保存到 import_log.txt")

if __name__ == "__main__":
    main()