import chromadb
from sentence_transformers import SentenceTransformer
import os
from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))

CHROMA_DIR = os.path.join(BASE_DIR, os.getenv("CHROMA_DIR", "chroma_db"))
MODEL_NAME = os.getenv("EMBED_MODEL", "paraphrase-multilingual-MiniLM-L12-v2")

embed_model = SentenceTransformer(MODEL_NAME)
client = chromadb.PersistentClient(path=CHROMA_DIR)
collection = client.get_collection('kg_contexts')

question = 'Why might researchers choose to model a DO as a particle-based interaction graph?'
vec = embed_model.encode([question]).tolist()
results = collection.query(query_embeddings=vec, n_results=8)

print('问题:', question)
print()
for i, (doc, meta, dist) in enumerate(zip(
        results['documents'][0],
        results['metadatas'][0],
        results['distances'][0])):
    print(f'[{i+1}] 相似度: {1-dist:.3f}')
    print(f'     {meta["subject"]} --[{meta["relation"]}]--> {meta["object"]}')
    print(f'     {doc[:100]}')
    print()
