"""
enrich_graph_metadata.py — post-build cleanup + metadata enrichment.

Runs idempotent Cypher migrations after the Layer 1 + Layer 2 indexers and
BEFORE the Layer 3 semantic extractor (so L3 entity extraction sees clean
content_raw without Markdown heading characters).

Property names align with Layer 1 indexer (`Article.art_num`, `Clause.clause_num`).
Layer 1 already populates these during structural indexing; this script only
fills any gaps and adds the cross-layer denormalizations needed for Hit@k.

What it does:
  1. Strips leading/embedded Markdown heading + bold characters from
     Clause.content_raw (`## `, `###`, `**`, etc.) — produced by the PDF
     parser when section headings were captured inline.
  2. Backfills Article.art_num from Article.title for any nodes missing it
     (defensive — Layer 1 already sets this).
  3. Backfills Clause.clause_num from the leading "N." in content_raw, using
     the Cypher DOTALL flag (defensive — Layer 1 already sets this).
  4. Denormalizes Article.id and Article.art_num onto each Clause as
     `article_id` and `article_num` so deterministic Hit@k checks
     (clause_id ∈ retrieved_ids) are one-line lookups instead of joins.
  5. Removes legacy/duplicate property names (`number`, `article_number`)
     left over from earlier migrations that introduced shadow fields.

Idempotent: re-running is a no-op if metadata is already populated.

Usage:
    python scripts/enrich_graph_metadata.py
    python scripts/enrich_graph_metadata.py --dry-run    # report only

Place this between L2 and L3 in the canonical pipeline:
    python -m src.indexers.layer1_structure
    python -m src.indexers.layer2_crossref
    python scripts/enrich_graph_metadata.py            # <-- here
    python -m src.indexers.layer3_semantics
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.core.config import SettingsConfig


CLEAN_MARKDOWN_CYPHER = """
MATCH (c:Clause)
WHERE c.content_raw IS NOT NULL
  AND (c.content_raw CONTAINS '#' OR c.content_raw CONTAINS '**')
WITH c, c.content_raw AS old
SET c.content_raw = trim(
    replace(replace(replace(replace(old, '###', ''), '##', ''), '#', ''), '**', '')
)
RETURN count(c) AS cleaned
"""

ARTICLE_NUMBER_CYPHER = """
MATCH (a:Article)
WHERE a.title =~ '^Article \\d+\\..*' AND a.art_num IS NULL
SET a.art_num = toInteger(split(replace(a.title, 'Article ', ''), '.')[0])
RETURN count(a) AS updated
"""

CLAUSE_NUMBER_CYPHER = """
MATCH (c:Clause)
WHERE c.content_raw IS NOT NULL AND c.clause_num IS NULL
WITH c, trim(c.content_raw) AS raw
WHERE raw =~ '(?s)^\\d+\\..*'
SET c.clause_num = toInteger(split(raw, '.')[0])
RETURN count(c) AS updated
"""

DENORMALIZE_CYPHER = """
MATCH (a:Article)-[:HAS_CLAUSE]->(c:Clause)
WHERE c.article_id IS NULL OR c.article_num IS NULL
SET c.article_num = a.art_num, c.article_id = a.id
RETURN count(c) AS updated
"""

# Drop legacy duplicate properties from earlier migrations that introduced
# `number`/`article_number` before the audit aligned on L1's `clause_num`/`art_num`.
DROP_LEGACY_CYPHER = """
MATCH (c:Clause)
REMOVE c.number, c.article_number
WITH c
MATCH (a:Article)
REMOVE a.number
RETURN 'legacy props removed' AS status
"""

VERIFY_CYPHER = """
MATCH (c:Clause)
RETURN count(c) AS total,
       count(c.clause_num) AS with_clause_num,
       count(c.article_num) AS with_article_num
"""


def run_step(session, name: str, cypher: str, dry_run: bool):
    if dry_run:
        print(f"[DRY-RUN] would run: {name}")
        return
    result = session.run(cypher).single()
    print(f"[{name}] {dict(result)}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Enrich graph metadata after L1+L2 indexing")
    parser.add_argument("--dry-run", action="store_true", help="Print steps without writing")
    args = parser.parse_args()

    driver = SettingsConfig.neo4j_driver

    print("=" * 70)
    print("GRAPH METADATA ENRICHMENT")
    print("=" * 70)

    with driver.session() as session:
        run_step(session, "1. Clean Markdown",       CLEAN_MARKDOWN_CYPHER,  args.dry_run)
        run_step(session, "2. Article.art_num",      ARTICLE_NUMBER_CYPHER,  args.dry_run)
        run_step(session, "3. Clause.clause_num",    CLAUSE_NUMBER_CYPHER,   args.dry_run)
        run_step(session, "4. Denormalize onto Clause", DENORMALIZE_CYPHER,  args.dry_run)
        run_step(session, "5. Drop legacy props",    DROP_LEGACY_CYPHER,     args.dry_run)

        if not args.dry_run:
            verify = session.run(VERIFY_CYPHER).single()
            d = dict(verify)
            print(f"\n[VERIFY] {d}")
            if d["total"] == d["with_clause_num"] == d["with_article_num"]:
                print("[PASS] All Clauses have number + article_number set.")
                return 0
            else:
                print("[WARN] Some Clauses missing metadata. Inspect:")
                missing = session.run("""
                    MATCH (c:Clause)
                    WHERE c.number IS NULL
                    RETURN c.id, left(c.content_raw, 80) AS preview
                    LIMIT 10
                """)
                for row in missing:
                    print(f"   {row['c.id']}: {row['preview']}")
                return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
