# GraphRAG for University Academic Regulations

**Pham Vu Bao — ITITWE20026 — BSc (Hons) Information Technology**

Question-answering over the IU Academic Regulations (QĐ.719) using a four-layer knowledge graph and Personalised PageRank retrieval.

---

## Prerequisites

- Python 3.10+
- [Neo4j Desktop](https://neo4j.com/download/) with a local DBMS running Neo4j 5.x
- A [Groq API key](https://console.groq.com/) (you will need more than 1 key for free tier 4 is sufficient)

---

## 1. Environment setup

```bash
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```
NEO4J_URI=neo4j://127.0.0.1:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password
GROQ_API_KEY=your_groq_api_key
```

---

## 2. Restore the graph database (recommended)

A pre-built database snapshot is included at `data/graph_backup/neo4j/`.
This is the exact graph used in the thesis experiment (117 clauses, 474 entities, 28 REFERS_TO edges, HNSW vector index).

**Steps:**

1. Open Neo4j Desktop and **stop** your DBMS.
2. Find your DBMS data folder. In Neo4j Desktop: click the three-dot menu on your DBMS → **Open Folder** → **Data**. Navigate to `databases/`.
3. Delete the existing `neo4j/` folder inside `databases/`.
4. Copy `data/graph_backup/neo4j/` from this project into `databases/`.
5. **Start** the DBMS.

Verify in Neo4j Browser (`http://localhost:7474`):

```cypher
MATCH (n) RETURN labels(n), count(n);
```

Expected: Document(1), Chapter(6), Article(22), Clause(117), Entity(474).

---

## 3. Build the graph from scratch (alternative)

Run the pipeline in order. Each step is idempotent.

```bash
# Layer 1: structural hierarchy (Document → Chapter → Article → Clause)
python -m src.indexers.layer1_structure

# Layer 2: LLM cross-reference edges
python -m src.indexers.layer2_crossref

# Between L2 and L3: clean markdown artifacts, backfill clause numbers
python scripts/enrich_graph_metadata.py

# Supplement Layer 2: regex REFERS_TO extraction (grows edges from ~12 to 28)
python scripts/extract_refers_to_regex.py

# Layer 3: LLM entity-relation extraction (requires GROQ_API_KEY)
python -m src.indexers.layer3_semantics

# Post-L3: deduplicate entities (497 raw → 474 canonical)
python scripts/deduplicate_entities.py

# Layer 4: HNSW vector index
python -m src.indexers.layer4_vector
```

Layer 3 uses `llama-3.3-70b-versatile` via Groq with automatic rate-limit fallback. Expect ~20–30 minutes on the free tier.

---

## 4. Run the demo app

```bash
streamlit run demo_app.py
```

Opens at `http://localhost:8501`. Three tabs: Live Q&A, Results Dashboard, Architecture.

---

## 5. Run the experiment

Runs all 28 questions across 3 retrieval modes (84 cells total). Run each mode separately so the demo app can load them independently:

```bash
python -m src.experiments.run_3x3 --mode VECTOR_RETRIEVAL --output outputs/eval_vec.csv
python -m src.experiments.run_3x3 --mode GRAPH_RETRIEVAL  --output outputs/eval_graph.csv
python -m src.experiments.run_3x3 --mode GRAPH_ITERATIVE  --output outputs/eval_iter.csv
```

Each command resumes automatically if interrupted (skips already-scored rows).

Then produce the summary tables used in the thesis:

```bash
python scripts/analyse_results.py
```

Output files:

| File | Contents |
|---|---|
| `outputs/eval_vec.csv` | Per-question results for VECTOR_RETRIEVAL |
| `outputs/eval_graph.csv` | Per-question results for GRAPH_RETRIEVAL |
| `outputs/eval_iter.csv` | Per-question results for GRAPH_ITERATIVE |
| `outputs/tables/overall_means.csv` | Mean BERTScore, ROUGE-L, Recall@3, latency per mode |
| `outputs/tables/per_type_means.csv` | Same, split by CROSSREF / MULTIHOP |
| `outputs/tables/wilcoxon.csv` | Wilcoxon signed-rank and paired t-test results |
| `outputs/tables/wilcoxon_stratified.csv` | Per question-type Wilcoxon (BERTScore F1, n=14) |
| `outputs/tables/retrieval.csv` | Hit@3, Recall@3, Precision@3 per mode and type |

---

## Project structure

```
src/
  indexers/         # Layer 1–4 graph builders
  engine/           # Retrieval modes (VECTOR, GRAPH, GRAPH_ITERATIVE)
  experiments/      # run_3x3.py experiment runner
  core/             # Config, LLM wrappers
scripts/
  enrich_graph_metadata.py   # Between L2 and L3: markdown clean + denormalize
  extract_refers_to_regex.py # Supplement L2: regex REFERS_TO extraction
  deduplicate_entities.py    # Post-L3: entity deduplication
data/
  raw/              # Source PDF (quy_dinh_dao_tao_EN.pdf)
  graph_backup/neo4j/  # Pre-built database snapshot
  test_set/         # 28-question evaluation set
outputs/
  tables/           # Experiment result CSVs
demo_app.py         # Streamlit demo
```
