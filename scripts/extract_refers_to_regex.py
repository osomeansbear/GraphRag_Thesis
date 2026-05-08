"""
Regex-based REFERS_TO edge extraction.

The existing LLM-based layer2_crossref.py produced only ~12 REFERS_TO edges.
This script supplements it by scanning every clause for explicit textual
references to other Articles/Clauses and creating REFERS_TO edges directly.

Patterns matched (case-insensitive):
  - "Article <N>" / "Articles <N> and <M>"
  - "Clause <N>.<M>" / "Point <N>.<M>"
  - "as stipulated in Article <N>"
  - "according to Article <N>"
  - "referred to in Clause <N>.<M>"
  - "provided in Article <N>, Clause <M>"

Run after building Layer 1 (structure). Does not require Layer 3 to be built.
Expected result: grows REFERS_TO from ~12 to 40+ edges.
"""

import os
import re
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.core.config import SettingsConfig

# --- Reference patterns ---
# The document uses REVERSED order: "clause N, Article M" (not "Article M, Clause N").
# Patterns verified against actual clause text samples.

# "clause 10, Article 2" / "clause 10, Article 2 of this Regulation"
CLAUSE_ARTICLE_PATTERN = re.compile(
    r"clause\s+(\d+),?\s+Article\s+(\d+)",
    re.IGNORECASE,
)

# "point d, clause 1, Article 16" — three-level reference
POINT_CLAUSE_ARTICLE_PATTERN = re.compile(
    r"point\s+[a-z],?\s+clause\s+(\d+),?\s+Article\s+(\d+)",
    re.IGNORECASE,
)

# "Article 9 of this Regulation" — article-only
ARTICLE_ONLY_PATTERN = re.compile(
    r"Article\s+(\d+)\s+of\s+this\s+Regulation",
    re.IGNORECASE,
)

# "clause 3 of this Article" — relative reference within the same article
RELATIVE_CLAUSE_PATTERN = re.compile(
    r"clause\s+(\d+)\s+of\s+this\s+[Aa]rticle",
    re.IGNORECASE,
)

# "as stipulated in Article N" / "according to Article N" — contextual article-only
ARTICLE_CONTEXTUAL_PATTERN = re.compile(
    r"(?:as stipulated in|according to|referred to in|provided in|under|pursuant to|in accordance with)\s+"
    r"Article\s+(\d+)",
    re.IGNORECASE,
)


def extract_references(text: str):
    """Return (absolute_refs, relative_clause_nums).

    absolute_refs: list of (article_num, clause_num_or_None)
    relative_clause_nums: list of clause_num ints from 'clause N of this Article'
    """
    refs = []
    relative = []

    for m in CLAUSE_ARTICLE_PATTERN.finditer(text):
        refs.append((int(m.group(2)), int(m.group(1))))

    for m in POINT_CLAUSE_ARTICLE_PATTERN.finditer(text):
        refs.append((int(m.group(2)), int(m.group(1))))

    for m in ARTICLE_ONLY_PATTERN.finditer(text):
        refs.append((int(m.group(1)), None))

    for m in ARTICLE_CONTEXTUAL_PATTERN.finditer(text):
        refs.append((int(m.group(1)), None))

    for m in RELATIVE_CLAUSE_PATTERN.finditer(text):
        relative.append(int(m.group(1)))

    return list(dict.fromkeys(refs)), list(dict.fromkeys(relative))


def build_clause_index(driver):
    """Return {(article_num, clause_num): clause_id} and {article_num: [clause_ids]}."""
    with driver.session() as session:
        result = session.run("""
            MATCH (c:Clause)
            RETURN c.id AS id, c.article_num AS article_num, c.clause_num AS clause_num
        """)
        clause_map = {}
        article_clauses = defaultdict(list)
        for rec in result:
            article_num = rec["article_num"]
            clause_num = rec["clause_num"]
            clause_id = rec["id"]
            if article_num is not None and clause_num is not None:
                clause_map[(int(article_num), int(clause_num))] = clause_id
            if article_num is not None:
                article_clauses[int(article_num)].append(clause_id)
    return clause_map, article_clauses


def resolve_reference(article_num, clause_num, clause_map, article_clauses):
    """Resolve a (article, clause_or_None) reference to a list of clause IDs."""
    if clause_num is not None:
        key = (article_num, clause_num)
        if key in clause_map:
            return [clause_map[key]]
        return []
    # Article-only reference: return first clause of that article
    clauses = article_clauses.get(article_num, [])
    return clauses[:1]  # anchor to first clause to avoid over-expansion


def main():
    driver = SettingsConfig.neo4j_driver

    print("Building clause index from Neo4j...")
    clause_map, article_clauses = build_clause_index(driver)
    print(f"Indexed {len(clause_map)} clauses across {len(article_clauses)} articles")

    print("Loading clause text...")
    with driver.session() as session:
        result = session.run("""
            MATCH (c:Clause)
            WHERE c.content_raw IS NOT NULL
            RETURN c.id AS id, c.content_raw AS text,
                   c.article_num AS article_num, c.clause_num AS clause_num
        """)
        clauses = [dict(rec) for rec in result]
    print(f"Scanning {len(clauses)} clauses for references...")

    edges_created = 0
    edges_skipped = 0

    with driver.session() as session:
        for clause in clauses:
            src_id = clause["id"]
            text = clause["text"] or ""
            src_article = clause["article_num"]
            abs_refs, rel_clause_nums = extract_references(text)

            # Absolute references (article + optional clause)
            for article_num, clause_num in abs_refs:
                target_ids = resolve_reference(article_num, clause_num, clause_map, article_clauses)
                for tgt_id in target_ids:
                    if tgt_id == src_id:
                        edges_skipped += 1
                        continue
                    res = session.run("""
                        MATCH (src:Clause {id: $src})
                        MATCH (tgt:Clause {id: $tgt})
                        MERGE (src)-[r:REFERS_TO]->(tgt)
                        ON CREATE SET r.source = 'regex', r.weight = 1.0
                        RETURN r.source AS src_field
                    """, src=src_id, tgt=tgt_id)
                    if res.single():
                        edges_created += 1

            # Relative references: "clause N of this Article" — resolve within same article
            if src_article is not None:
                for rel_clause_num in rel_clause_nums:
                    key = (int(src_article), rel_clause_num)
                    tgt_id = clause_map.get(key)
                    if not tgt_id or tgt_id == src_id:
                        edges_skipped += 1
                        continue
                    res = session.run("""
                        MATCH (src:Clause {id: $src})
                        MATCH (tgt:Clause {id: $tgt})
                        MERGE (src)-[r:REFERS_TO]->(tgt)
                        ON CREATE SET r.source = 'regex_relative', r.weight = 1.0
                        RETURN r.source AS src_field
                    """, src=src_id, tgt=tgt_id)
                    if res.single():
                        edges_created += 1

    print(f"\nDone.")
    print(f"  REFERS_TO edges created/merged: {edges_created}")
    print(f"  Self-references skipped:         {edges_skipped}")
    print("Run check_layer3.py to verify the updated graph.")


if __name__ == "__main__":
    main()
