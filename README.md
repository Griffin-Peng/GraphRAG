# GraphRAG 知识问答系统

基于知识图谱增强检索（GraphRAG）的 AI/CS 领域知识问答系统，覆盖 200 篇学术论文。

## 架构

```
用户问题
   ├── 向量检索 (ChromaDB) ──→ 语义相关三元组
   └── 图谱检索 (Neo4j) ────→ 1-2 跳关系路径扩展
                ↓
         拼接上下文 → Qwen-Plus 生成答案
```

- **Neo4j** — 知识图谱存储，17,741 节点
- **ChromaDB** — 向量相似度检索，21,534 条上下文
- **SentenceTransformer** — `paraphrase-multilingual-MiniLM-L12-v2` 向量化
- **Qwen-Plus** — 通义千问 LLM（DashScope API）

## 快速开始

### 前置要求

- Python 3.10+
- Neo4j 数据库（运行中）
- DashScope API Key

### 安装

```bash
git clone <repo-url>
cd GraphRAG
pip install -r requirements.txt
```

### 配置

```bash
cp .env.example .env
# 编辑 .env 填入你的 API Key 和 Neo4j 密码
```

### 构建知识库（首次使用）

```bash
# 1. 解析 TTL 文件并导入 Neo4j
python import_to_neo4j.py

# 2. 构建向量索引
python build_vector_store.py
```

### 启动 Web 界面

```bash
streamlit run app.py
```

打开 http://localhost:8501 即可使用。

### 命令行问答

```bash
python graphrag.py
```

## 三种检索模式

| 模式 | 说明 |
|------|------|
| **Plain** | 直接询问 LLM，无额外上下文 |
| **RAG** | 仅使用向量检索的语义匹配结果 |
| **GraphRAG** | 向量检索 + 图谱路径扩展（1-2 跳） |

## 评测结果

在 750 道 one-hop / two-hop / multi-hop 单选题上的准确率：

| 方法 | One-Hop | Two-Hop | Multi-Hop |
|------|---------|---------|-----------|
| Plain | 94.0% | 90.8% | 89.2% |
| RAG | 95.6% | 93.6% | 89.6% |
| **GraphRAG** | **96.0%** | **94.0%** | **90.0%** |

GraphRAG 在多跳推理上优势显著，multi-hop 提升达 +10.4%（相比无阈值过滤版本）。

### 运行评测

```bash
python evaluate.py
```

结果保存至 `eval_results.json`。

## 项目结构

```
GraphRAG/
├── app.py                 # Streamlit Web 界面
├── graphrag.py            # 命令行交互式问答
├── evaluate.py            # 三种方法对比评测
├── parse_ttl.py           # TTL 知识图谱解析
├── import_to_neo4j.py     # 三元组导入 Neo4j
├── build_vector_store.py  # 构建 ChromaDB 向量库
├── debug.py               # 检索调试工具
├── KG_TTL/                # 200 篇论文的 TTL 知识图谱
├── hop_qa/                # QA 评测数据集
├── requirements.txt
└── .env.example
```

## License

MIT
