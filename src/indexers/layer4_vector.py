import time
from src.core.config import SettingsConfig
from tqdm import tqdm

def build_layer_4_vector_index():
    print("--- LAYER 4: VECTOR EMBEDDING & INDEXING ---")

    store = SettingsConfig.get_graph_store()

    # 1. Fetch all Clauses (including any that failed semantic extraction)
    print("Loading Clause list...")
    results = store.client.execute_query("""
        MATCH (c:Clause)
        WHERE c.content_raw IS NOT NULL
        RETURN c.id AS id, c.content_raw AS text, c.embedding AS existing_emb
    """)

    # Filter out nodes that already have embeddings (safe to re-run)
    nodes_to_embed = [
        {"id": r["id"], "text": r["text"]}
        for r in results.records
        if r["existing_emb"] is None
    ]

    total = len(results.records)
    missing = len(nodes_to_embed)
    print(f"Total Clauses: {total}. Needing embeddings: {missing}.")

    if missing > 0:
        print("Computing embeddings (local CPU: all-MiniLM-L6-v2)...")

        batch_size = 64
        for i in tqdm(range(0, missing, batch_size), desc="Embedding"):
            batch = nodes_to_embed[i : i + batch_size]
            texts = [n["text"] for n in batch]

            try:
                embeddings = SettingsConfig.embed_model.get_text_embedding_batch(texts)

                for j, node in enumerate(batch):
                    vector = embeddings[j]
                    store.client.execute_query("""
                        MATCH (c:Clause {id: $id})
                        SET c.embedding = $vector
                    """, id=node["id"], vector=vector)
            except Exception as e:
                print(f"Error in batch at {i}: {e}")
                continue

    # 2. Create vector index in Neo4j
    print("Configuring vector index in Neo4j...")

    index_name = "clause_vector_index"

    check = store.client.execute_query("SHOW INDEXES WHERE name = $name", name=index_name)

    if len(check.records) == 0:
        store.client.execute_query("""
            CREATE VECTOR INDEX clause_vector_index IF NOT EXISTS
            FOR (c:Clause)
            ON (c.embedding)
            OPTIONS {indexConfig: {
             `vector.dimensions`: 384,
             `vector.similarity_function`: 'cosine'
            }}
        """)
        print(f"Created new index: {index_name}")
    else:
        print(f"Index '{index_name}' already exists.")

    # 3. Quick smoke test
    print("Waiting for Neo4j to finish indexing (5s)...")
    time.sleep(5)

    print("\n--- Quick test: 'When is a student expelled?' ---")
    test_query = "When is a student expelled from the program"

    q_vector = SettingsConfig.embed_model.get_query_embedding(test_query)

    search_res = store.client.execute_query("""
        CALL db.index.vector.queryNodes('clause_vector_index', 3, $vec)
        YIELD node, score
        RETURN node.content_raw AS content, score
    """, vec=q_vector)

    for r in search_res.records:
        print(f"   Score: {r['score']:.4f}")
        print(f"   Text: {r['content'][:150]}...")

    print("\nDone. Layer 4 complete.")

if __name__ == "__main__":
    build_layer_4_vector_index()
