from rdflib import Graph, URIRef
from rdflib.namespace import RDF
import os
import re
from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))

def extract_ttl_content(filepath):
    """从混有Markdown注释的文件中提取所有turtle代码块内容"""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # 匹配所有 ```turtle ... ``` 代码块
    blocks = re.findall(r'```turtle\s*(.*?)```', content, re.DOTALL)

    if not blocks:
        print(f"警告：{filepath} 中未找到turtle代码块")
        return None

    # 合并所有块，加上命名空间前缀（rdflib解析必须有）
    prefix = "@prefix : <http://example.org/> .\n@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .\n\n"
    #print("prefix:\n" + prefix + '\n'.join(blocks)[:500])  # 打印前500字符预览
    return prefix + '\n'.join(blocks)


def parse_triples(filepath):
    """解析单个TTL文件，返回三元组列表"""
    ttl_content = extract_ttl_content(filepath)
    #print("ttl_content:\n" + (ttl_content[:500] if ttl_content else "None"))  # 打印前500字符预览
    if ttl_content is None:
        return []

    g = Graph()
    try:
        g.parse(data=ttl_content, format="turtle")
    except Exception as e:
        print(f"解析失败 {filepath}: {e}")
        return []

    triples = []
    for s, p, o in g:
        # 只要两端都是实体的关系（跳过 rdf:type 和属性）
        if isinstance(o, URIRef) and p != RDF.type:
            s_name = str(s).split("/")[-1]
            p_name = str(p).split("/")[-1]
            o_name = str(o).split("/")[-1]

            # 跳过 sourceChunk、sourceSection 等元数据关系
            if p_name in ['sourceChunk', 'sourceSection', 'contextText']:
                continue

            triples.append({
                "subject": s_name,
                "relation": p_name,
                "object": o_name
            })

    return triples


# ===== 测试单个文件 =====
ttl_dir = os.path.join(BASE_DIR, os.getenv("KG_TTL_DIR", "KG_TTL"))
files = [f for f in os.listdir(ttl_dir) if f.endswith('.ttl')]
print(f"共找到 {len(files)} 个TTL文件")

test_file = os.path.join(ttl_dir, files[0])
print(f"\n测试文件：{files[0]}")

triples = parse_triples(test_file)
print(f"解析到 {len(triples)} 条关系三元组\n")

print("前20条：")
for t in triples[:20]:
    print(f"  {t['subject']} --[{t['relation']}]--> {t['object']}")