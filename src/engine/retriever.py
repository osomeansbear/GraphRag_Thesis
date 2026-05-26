"""
ThesisRetriever — three-mode retrieval engine for thesis evaluation.

Modes:
  - VECTOR_RETRIEVAL:   dense vector search only. No graph traversal. Baseline.
  - GRAPH_RETRIEVAL:    HippoRAG-style PPR over L2+L3 combined graph with
                        query-entity seeding and CrossEncoder post-expansion filter.
                        Thesis proposed method.
  - GRAPH_ITERATIVE:    IRCoT-style one-cycle iterative extension of GRAPH_RETRIEVAL.
                        Retrieve -> Reflect (LLM) -> Retrieve again on gap -> Merge.

Components:
  - Sentence-transformer embedding (all-MiniLM-L6-v2): local, no API cost.
  - Personalized PageRank via networkx: alpha=0.85, edge weights causal=3,
    conditional=2 (default=1.0). Seeds: vector anchor clauses + query entities.
  - CrossEncoder post-expansion filter (ms-marco-MiniLM-L-6-v2): eliminates
    context-polluting clauses before passing to the LLM.
  - Reflection LLM: structured JSON audit call used only by
    GRAPH_ITERATIVE to identify missing regulatory concepts after Retrieve-1.
"""

import re
from typing import List, Dict, Tuple
from sentence_transformers import CrossEncoder
from llama_index.core import Settings
from src.core.config import SettingsConfig

# Cross-encoder model for relevance scoring (loaded once at module level)
_CROSS_ENCODER = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# ------------------------------------------------------------------
# Edge weight tiers — pattern-matched against any relationship name.
# This handles both known schema edges AND dynamically generated ones.
# ------------------------------------------------------------------
CAUSAL_PATTERNS      = re.compile(r'REQUIRES|LEADS_TO', re.I)
CONDITIONAL_PATTERNS = re.compile(r'AFFECTS|HAS_CONDITION|EXEMPTS', re.I)
# Everything else (PERFORMS, MENTIONS, etc.) → default weight

EDGE_TIER_WEIGHTS = {
    "causal":      3.0,
    "conditional": 2.0,
    "default":     1.0,
}

# Minimum vector similarity to pass initial retrieval (raised from 0.3)
VECTOR_SCORE_THRESHOLD = 0.5


def _edge_weight(rel_name: str) -> float:
    """Return the semantic weight for any relationship type name."""
    if CAUSAL_PATTERNS.search(rel_name):
        return EDGE_TIER_WEIGHTS["causal"]
    if CONDITIONAL_PATTERNS.search(rel_name):
        return EDGE_TIER_WEIGHTS["conditional"]
    return EDGE_TIER_WEIGHTS["default"]



class ThesisRetriever:
    def __init__(self):
        self.driver = SettingsConfig.neo4j_driver
        self.embed_model = SettingsConfig.embed_model
        self.llm = None

    # ------------------------------------------------------------------
    # Cross-Encoder Re-ranker
    # ------------------------------------------------------------------
    def _rerank(self, query: str, chunks: List[Dict], top_n: int = 3) -> List[Dict]:
        """
        Score each chunk for relevance using a dedicated cross-encoder model
        (cross-encoder/ms-marco-MiniLM-L-6-v2). Deterministic, <100ms, no API quota.
        Returns top_n chunks sorted by relevance score (descending).
        """
        if not chunks:
            return chunks

        pairs = [(query, chunk["text"][:512]) for chunk in chunks]
        scores = _CROSS_ENCODER.predict(pairs)

        scored = [
            {**chunk, "rerank_score": float(score)}
            for chunk, score in zip(chunks, scores)
        ]
        scored.sort(key=lambda x: x["rerank_score"], reverse=True)
        return scored[:top_n]

    # ------------------------------------------------------------------
    # Core vector search
    # ------------------------------------------------------------------
    def _vector_search(self, query: str, top_k: int) -> List[Dict]:
        """
        Embed the raw query and run ANN search.
        Applies VECTOR_SCORE_THRESHOLD to filter low-quality matches.
        """
        query_vector = self.embed_model.get_query_embedding(query)
        try:
            with self.driver.session() as session:
                result = session.run("""
                    CALL db.index.vector.queryNodes('clause_vector_index', $k, $vec)
                    YIELD node, score
                    WHERE score > $threshold
                    RETURN node.id as id,
                           node.content_raw as text,
                           node.article_num as article_num,
                           node.clause_num as clause_num,
                           score
                """, k=top_k, vec=query_vector, threshold=VECTOR_SCORE_THRESHOLD)
                return [record.data() for record in result]
        except Exception as e:
            print(f"WARNING: Vector Search Error: {e}")
            return []

    # ------------------------------------------------------------------
    # Confidence helper
    # ------------------------------------------------------------------
    @staticmethod
    def _confidence(chunks: List[Dict]) -> Tuple[float, str]:
        """Derive a confidence label from mean vector similarity score."""
        if not chunks:
            return 0.0, "NONE"
        mean = sum(c["score"] for c in chunks) / len(chunks)
        if mean >= 0.75:
            return mean, "HIGH"
        if mean >= 0.55:
            return mean, "MEDIUM"
        return mean, "LOW"

    # ------------------------------------------------------------------
    # VECTOR_RETRIEVAL — dense vector baseline (no graph, no reranker)
    # ------------------------------------------------------------------
    def retrieve_vector(self, query: str, top_k: int = 5) -> Tuple[str, str, list, list]:
        """
        Dense vector search with no graph traversal and no reranker.
        Baseline method for the thesis 2-mode comparison.
        Returns (context_str, confidence, chunks, retrieved_keys).
        """
        query_vector = self.embed_model.get_query_embedding(query)
        try:
            with self.driver.session() as session:
                result = session.run("""
                    CALL db.index.vector.queryNodes('clause_vector_index', $k, $vec)
                    YIELD node, score
                    WHERE score > $threshold
                    RETURN node.id as id,
                           node.content_raw as text,
                           node.article_num as article_num,
                           node.clause_num as clause_num,
                           score
                """, k=top_k, vec=query_vector, threshold=VECTOR_SCORE_THRESHOLD)
                nodes = [record.data() for record in result]
        except Exception as e:
            print(f"WARNING: Vector Search Error: {e}")
            return "", "NONE", [], []

        top3 = sorted(nodes, key=lambda x: x["score"], reverse=True)[:3]
        _, confidence = self._confidence(top3)
        chunks = [f"[Vector: {n['score']:.2f}]\n{n['text']}" for n in top3]
        keys = [(n.get("article_num"), n.get("clause_num")) for n in top3]
        return "\n\n".join(chunks), confidence, chunks, keys

    # ------------------------------------------------------------------
    # GRAPH_RETRIEVAL helpers
    # ------------------------------------------------------------------
    def _build_ppr_graph(self, session):
        """Pull the full entity+clause subgraph from Neo4j into a networkx DiGraph.

        Node types: Clause (type='Clause'), Entity (type='Entity').
        Edge weights use the same tier as path scoring:
          causal (REQUIRES/LEADS_TO) = 3.0, conditional = 2.0,
          REFERS_TO (clause-clause L2) = 2.0, MENTIONS (clause-entity) = 1.0.
        All edges added bidirectionally so PPR can diffuse in both directions.
        """
        import networkx as nx
        G = nx.DiGraph()

        # Entity-entity typed relations (L3)
        res = session.run("""
            MATCH (e1:Entity)-[r]->(e2:Entity)
            RETURN e1.name AS src, type(r) AS rel_type, e2.name AS tgt,
                   coalesce(r.weight, 1.0) AS stored_weight
        """)
        for rec in res:
            w = _edge_weight(rec["rel_type"])
            G.add_node(rec["src"], node_type="Entity")
            G.add_node(rec["tgt"], node_type="Entity")
            G.add_edge(rec["src"], rec["tgt"], weight=w)
            G.add_edge(rec["tgt"], rec["src"], weight=w * 0.5)  # weak reverse

        # MENTIONS edges: Clause -> Entity (bidirectional for diffusion)
        res = session.run("""
            MATCH (c:Clause)-[:MENTIONS]->(e:Entity)
            RETURN c.id AS clause_id, c.article_num AS an, c.clause_num AS cn,
                   e.name AS entity_name
        """)
        for rec in res:
            cid = rec["clause_id"]
            ename = rec["entity_name"]
            G.add_node(cid, node_type="Clause",
                       article_num=rec["an"], clause_num=rec["cn"])
            G.add_node(ename, node_type="Entity")
            G.add_edge(cid, ename, weight=1.0)
            G.add_edge(ename, cid, weight=0.5)  # entity -> clause (weaker)

        # REFERS_TO edges: Clause -> Clause (L2, weight=2.0, bidirectional)
        res = session.run("""
            MATCH (c1:Clause)-[:REFERS_TO]-(c2:Clause)
            RETURN DISTINCT c1.id AS src, c2.id AS tgt,
                   c1.article_num AS an1, c1.clause_num AS cn1,
                   c2.article_num AS an2, c2.clause_num AS cn2
        """)
        for rec in res:
            G.add_node(rec["src"], node_type="Clause",
                       article_num=rec["an1"], clause_num=rec["cn1"])
            G.add_node(rec["tgt"], node_type="Clause",
                       article_num=rec["an2"], clause_num=rec["cn2"])
            G.add_edge(rec["src"], rec["tgt"], weight=2.0)
            G.add_edge(rec["tgt"], rec["src"], weight=2.0)

        return G

    def _extract_query_entities(self, query: str, session) -> List[str]:
        """Match query tokens against Entity node names for PPR seeding (fix 2.3).

        Uses two strategies:
          1. Full name match (case-insensitive) — highest precision.
          2. Longest-word match: any entity name word >3 chars in query.
        Returns up to 10 matched entity names to avoid over-seeding.
        """
        res = session.run("MATCH (e:Entity) RETURN e.name AS name")
        all_names = [rec["name"] for rec in res if rec["name"]]

        query_lower = query.lower()
        query_words = set(w.lower() for w in re.findall(r'\w+', query) if len(w) > 3)

        matched = []
        for name in all_names:
            name_lower = name.lower()
            if name_lower in query_lower:
                matched.append(name)
                continue
            name_words = set(w.lower() for w in name.split() if len(w) > 3)
            if name_words and name_words.issubset(query_words):
                matched.append(name)

        return matched[:10]

    def _rerank_filter(self, query: str, candidates: List[Dict],
                       top_n: int = 5) -> List[Dict]:
        """CrossEncoder filter after PPR expansion — ranks by relevance, returns top_n.

        No score threshold: CrossEncoder logits from ms-marco-MiniLM can be negative
        for scenario-based queries against regulatory text, which would incorrectly
        produce empty results. Always returns the top_n highest-scoring candidates.
        """
        if not candidates:
            return []
        pairs = [(query, (c.get("text") or "")[:512]) for c in candidates]
        scores = _CROSS_ENCODER.predict(pairs)
        for c, s in zip(candidates, scores):
            c["rerank_score"] = float(s)
        candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
        return candidates[:top_n]

    # ------------------------------------------------------------------
    # GRAPH_RETRIEVAL — PPR over L2+L3 + query-entity seeding + CrossEncoder filter
    # ------------------------------------------------------------------
    def retrieve_graph(self, query: str, top_k: int = 5, trace: dict = None) -> Tuple[str, str, list, list]:
        """
        Graph retrieval: Personalized PageRank over the combined L2+L3 graph,
        seeded by vector anchor clauses (70%) + query entity nodes (30%), with
        CrossEncoder post-expansion filtering to remove context-polluting clauses.

        Returns (context_str, confidence, chunks, retrieved_keys).
        """
        import networkx as nx

        anchor_weight = 0.7
        entity_weight = 0.3

        anchors = self._vector_search(query, top_k)
        if not anchors:
            return "", "NONE", [], []

        _, confidence = self._confidence(anchors)
        anchor_ids = [n["id"] for n in anchors]

        with self.driver.session() as session:
            G = self._build_ppr_graph(session)

            if len(G.nodes()) == 0:
                return self.retrieve_vector(query, top_k)

            # --- Personalization vector ---
            personalization: Dict[str, float] = {node: 0.0 for node in G.nodes()}

            valid_anchors = [cid for cid in anchor_ids if cid in personalization]
            if valid_anchors:
                per_anchor = anchor_weight / len(valid_anchors)
                for cid in valid_anchors:
                    personalization[cid] += per_anchor

            query_entities = self._extract_query_entities(query, session)
            valid_entities = [e for e in query_entities if e in personalization]
            if valid_entities and entity_weight > 0:
                per_entity = entity_weight / len(valid_entities)
                for ename in valid_entities:
                    personalization[ename] += per_entity

            # Normalize to sum=1 (required by networkx pagerank)
            total = sum(personalization.values())
            if total > 0:
                personalization = {k: v / total for k, v in personalization.items()}
            else:
                personalization = None  # falls back to uniform

            # --- Run PPR ---
            try:
                ppr_scores = nx.pagerank(
                    G, alpha=0.85,
                    personalization=personalization,
                    weight="weight",
                    max_iter=200,
                )
            except nx.PowerIterationFailedConvergence:
                ppr_scores = nx.pagerank(G, alpha=0.85, personalization=personalization,
                                         weight="weight", max_iter=500, tol=1e-4)

            # Top expanded Clause nodes (exclude anchors)
            anchor_set = set(anchor_ids)
            clause_scores = {
                node: score
                for node, score in ppr_scores.items()
                if G.nodes[node].get("node_type") == "Clause"
                and node not in anchor_set
            }
            top_expanded_ids = sorted(
                clause_scores, key=clause_scores.get, reverse=True
            )[:15]

            # Fetch clause text for all candidates
            all_ids = anchor_ids + top_expanded_ids
            res = session.run("""
                MATCH (c:Clause)
                WHERE c.id IN $ids
                RETURN c.id AS id, c.content_raw AS text,
                       c.article_num AS article_num, c.clause_num AS clause_num
            """, ids=all_ids)
            id_to_clause = {rec["id"]: dict(rec) for rec in res}

        # Build candidate list
        candidates = []
        for n in anchors:
            clause = id_to_clause.get(n["id"], {})
            candidates.append({
                "id": n["id"],
                "text": clause.get("text") or n.get("text", ""),
                "article_num": clause.get("article_num"),
                "clause_num": clause.get("clause_num"),
                "score": n["score"],
                "source": "anchor",
            })
        for cid in top_expanded_ids:
            clause = id_to_clause.get(cid, {})
            candidates.append({
                "id": cid,
                "text": clause.get("text", ""),
                "article_num": clause.get("article_num"),
                "clause_num": clause.get("clause_num"),
                "score": clause_scores.get(cid, 0.0),
                "source": "ppr",
            })

        # CrossEncoder filter - kill context pollution
        # top_n=3 matches VECTOR_RETRIEVAL's 3-clause context window for fair comparison.
        candidates = self._rerank_filter(query, candidates, top_n=3)

        # --- Optional retrieval trace for the demo. When trace is None (every
        #     run_3x3.py / experiment call) this block is skipped and behaviour
        #     is unchanged. ---
        if trace is not None:
            trace["query"] = query
            trace["anchors"] = [
                {"id": n["id"], "article_num": n.get("article_num"),
                 "clause_num": n.get("clause_num"), "score": float(n["score"])}
                for n in anchors
            ]
            trace["entity_seeds"] = list(valid_entities)
            trace["ppr_expanded"] = [
                {"id": cid,
                 "article_num": id_to_clause.get(cid, {}).get("article_num"),
                 "clause_num": id_to_clause.get(cid, {}).get("clause_num"),
                 "ppr_score": float(clause_scores.get(cid, 0.0))}
                for cid in top_expanded_ids
            ]
            trace["final"] = [
                {"id": c["id"], "article_num": c.get("article_num"),
                 "clause_num": c.get("clause_num"),
                 "rerank_score": float(c.get("rerank_score", 0.0)),
                 "source": c.get("source")}
                for c in candidates
            ]
            trace["graph"] = G

        chunks = [
            f"[{c['source'].upper()} score={c['score']:.3f} | relevance={c.get('rerank_score', 0.0):.2f}]\n{c['text']}"
            for c in candidates
        ]
        keys = [(c.get("article_num"), c.get("clause_num")) for c in candidates]
        return "\n\n".join(chunks), confidence, chunks, keys

    # ------------------------------------------------------------------
    # Reflection helper for GRAPH_ITERATIVE
    # ------------------------------------------------------------------

    _REFLECTION_PROMPT = (
        "Identify the single most important regulatory concept, clause, or definition "
        "that is NOT covered in the retrieved context but would be needed to fully answer "
        "the question — including any grading tables, classification bands, numerical "
        "thresholds, or definitional articles explicitly required by the question.\n\n"
        "QUESTION: {query}\n\n"
        "RETRIEVED CONTEXT:\n{context}\n\n"
        "Output strict JSON only. No other text.\n"
        "If the context is already fully sufficient with no missing definitions or thresholds:\n"
        '  {{"gap": null}}\n'
        "Otherwise output the one most important missing concept as a short search phrase:\n"
        '  {{"gap": "<specific regulatory concept, threshold table, or article section missing>"}}'
    )

    _THINK_RE_REFLECT = re.compile(
        r'<\s*think\s*>.*?<\s*/\s*think\s*>', re.IGNORECASE | re.DOTALL
    )

    def _reflect_on_context(self, query: str, context_str: str,
                            llm=None) -> "str | None":
        """
        Structured LLM call: identify the most important missing regulatory concept.
        Returns the gap phrase if something is missing, None if truly complete.
        Failure always returns None — no regression on error.

        llm: LLM instance to use. Falls back to self._reflect_llm if set,
             then initialises SingleGroqLLM(llama-3.1-8b-instant).
        """
        import json
        from llama_index.core.llms import ChatMessage

        reflect_llm = llm or getattr(self, "_reflect_llm", None)
        if reflect_llm is None:
            from src.core.single_groq_llm import SingleGroqLLM
            reflect_llm = SingleGroqLLM(model="llama-3.1-8b-instant")
            self._reflect_llm = reflect_llm

        prompt = self._REFLECTION_PROMPT.format(
            query=query,
            context=context_str[:2000],
        )
        try:
            response = reflect_llm.chat(
                [ChatMessage(role="user", content=prompt)]
            )
            text = self._THINK_RE_REFLECT.sub("", str(response)).strip()
            text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
            text = re.sub(r"^[^{\[]*", "", text).strip()  # strip "assistant: " prefix
            data = json.loads(text)
            gap = data.get("gap")
            if gap:
                return str(gap).strip() or None
        except Exception:
            pass
        return None

    # ------------------------------------------------------------------
    # GRAPH_ITERATIVE — one-cycle IRCoT-style iterative retrieval
    # ------------------------------------------------------------------

    def retrieve_graph_iterative(
        self, query: str, top_k: int = 5, trace: dict = None
    ) -> Tuple[str, str, list, list]:
        """
        One-cycle iterative retrieval over the L2+L3 graph.

        Step 1 (Retrieve-1): GRAPH_RETRIEVAL(query)          -> C1
        Step 2 (Reflect):    LLM audit: ANSWERABLE | MISSING + gap
        Step 3 (Retrieve-2): if MISSING, GRAPH_RETRIEVAL(gap) -> C2
        Step 4 (Merge):      dedup(C1 + C2)[:8] as final context

        The gap string (or None) is stored in self.last_gap for CSV logging.
        Returns (context_str, confidence, chunks, retrieved_keys).
        """
        self.last_gap = None

        ctx1, conf1, chunks1, keys1 = self.retrieve_graph(query, top_k, trace=trace)
        if not ctx1:
            return ctx1, conf1, chunks1, keys1

        gap = self._reflect_on_context(query, ctx1)
        self.last_gap = gap

        if gap is None:
            return ctx1, conf1, chunks1, keys1

        _, _, chunks2, keys2 = self.retrieve_graph(gap, top_k)

        seen: set = set()
        merged_chunks: list = []
        merged_keys: list = []

        for chunk, key in zip(chunks1, keys1):
            k = (key[0], key[1]) if key else (None, None)
            if k not in seen:
                seen.add(k)
                merged_chunks.append(chunk)
                merged_keys.append(key)

        for chunk, key in zip(chunks2, keys2):
            k = (key[0], key[1]) if key else (None, None)
            if k not in seen:
                seen.add(k)
                merged_chunks.append(chunk)
                merged_keys.append(key)

        merged_chunks = merged_chunks[:8]
        merged_keys = merged_keys[:8]

        return "\n\n".join(merged_chunks), conf1, merged_chunks, merged_keys

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------
    def search(self, query: str, mode: str) -> Tuple[str, str, list, list]:
        """
        Returns (context_string, confidence_label, chunks, retrieved_keys).
        confidence_label: 'HIGH' | 'MEDIUM' | 'LOW' | 'NONE'
        chunks: list of individual context strings
        retrieved_keys: list of (article_num, clause_num) tuples used by Hit@k.

        Modes:
          VECTOR_RETRIEVAL → dense vector baseline (no graph, no reranker)
          GRAPH_RETRIEVAL  → PPR over L2+L3 + query-entity seeding + CrossEncoder filter
          GRAPH_ITERATIVE  → GRAPH_RETRIEVAL + one IRCoT reflection cycle
        """
        if mode == "VECTOR_RETRIEVAL":
            return self.retrieve_vector(query)
        if mode == "GRAPH_RETRIEVAL":
            return self.retrieve_graph(query)
        if mode == "GRAPH_ITERATIVE":
            return self.retrieve_graph_iterative(query)
        return "", "NONE", [], []
