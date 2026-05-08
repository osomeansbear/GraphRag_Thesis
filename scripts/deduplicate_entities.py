"""
Entity deduplication for the Graph RAG knowledge base.

Pulls all Entity nodes, embeds their names with the same sentence-transformer
already used by the pipeline (all-MiniLM-L6-v2), clusters by cosine similarity
at a configurable threshold, then merges duplicates into a single canonical node
by redirecting all incident edges and deleting the duplicates.

Run AFTER re-building Layer 3 and BEFORE re-running evaluation.
"""

import os
import sys
import argparse
import numpy as np
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.core.config import SettingsConfig


def load_entities(driver):
    with driver.session() as session:
        result = session.run("MATCH (e:Entity) RETURN e.name AS name")
        return [rec["name"] for rec in result if rec["name"]]


def embed_names(names, embed_model):
    return embed_model.get_text_embedding_batch(names, show_progress=True)


def cosine_similarity(a, b):
    a, b = np.array(a), np.array(b)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 0 else 0.0


def cluster_entities(names, embeddings, threshold=0.85):
    """Single-pass greedy clustering: first name in each cluster becomes the seed."""
    n = len(names)
    assigned = [-1] * n
    clusters = []

    for i in range(n):
        if assigned[i] != -1:
            continue
        cluster_idx = len(clusters)
        clusters.append([i])
        assigned[i] = cluster_idx
        for j in range(i + 1, n):
            if assigned[j] != -1:
                continue
            sim = cosine_similarity(embeddings[i], embeddings[j])
            if sim >= threshold:
                clusters[cluster_idx].append(j)
                assigned[j] = cluster_idx

    return clusters


def pick_canonical(names, indices):
    """Pick the shortest name in the cluster as the canonical form."""
    cluster_names = [(names[i], i) for i in indices]
    cluster_names.sort(key=lambda x: len(x[0]))
    return cluster_names[0][0]


def merge_cluster(driver, canonical_name, duplicate_names, dry_run=False):
    """Redirect all edges from duplicates to canonical node, then delete duplicates."""
    relation_types_result = []
    with driver.session() as session:
        res = session.run("""
            MATCH ()-[r]->()
            RETURN DISTINCT type(r) AS rel_type
        """)
        relation_types_result = [rec["rel_type"] for rec in res]

    for dup_name in duplicate_names:
        if dup_name == canonical_name:
            continue
        print(f"  Merging '{dup_name}' -> '{canonical_name}'")
        if dry_run:
            continue
        with driver.session() as session:
            # Redirect MENTIONS edges (Clause -> Entity)
            session.run("""
                MATCH (c:Clause)-[m:MENTIONS]->(dup:Entity {name: $dup})
                MATCH (canonical:Entity {name: $canon})
                MERGE (c)-[:MENTIONS]->(canonical)
                DELETE m
            """, dup=dup_name, canon=canonical_name)

            # Redirect outgoing typed relations from duplicate
            for rel_type in relation_types_result:
                if rel_type in ("MENTIONS",):
                    continue
                session.run(f"""
                    MATCH (dup:Entity {{name: $dup}})-[r:`{rel_type}`]->(other)
                    MATCH (canonical:Entity {{name: $canon}})
                    WHERE other.name <> $canon
                    MERGE (canonical)-[:`{rel_type}` {{weight: coalesce(r.weight, 1.0), evidence: coalesce(r.evidence, '')}}]->(other)
                    DELETE r
                """, dup=dup_name, canon=canonical_name)

                # Redirect incoming typed relations to duplicate
                session.run(f"""
                    MATCH (other)-[r:`{rel_type}`]->(dup:Entity {{name: $dup}})
                    MATCH (canonical:Entity {{name: $canon}})
                    WHERE other.name <> $canon
                    MERGE (other)-[:`{rel_type}` {{weight: coalesce(r.weight, 1.0), evidence: coalesce(r.evidence, '')}}]->(canonical)
                    DELETE r
                """, dup=dup_name, canon=canonical_name)

            # Delete the now-isolated duplicate node
            session.run("""
                MATCH (dup:Entity {name: $dup})
                DETACH DELETE dup
            """, dup=dup_name)


def main():
    parser = argparse.ArgumentParser(description="Deduplicate Entity nodes in the graph")
    parser.add_argument("--threshold", type=float, default=0.85,
                        help="Cosine similarity threshold for clustering (default: 0.85)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print planned merges without modifying the graph")
    args = parser.parse_args()

    driver = SettingsConfig.neo4j_driver
    embed_model = SettingsConfig.embed_model

    print("Loading entity names from Neo4j...")
    names = load_entities(driver)
    print(f"Found {len(names)} Entity nodes")

    if not names:
        print("No entities found. Exiting.")
        return

    print("Embedding entity names...")
    embeddings = embed_names(names, embed_model)

    print(f"Clustering with threshold={args.threshold}...")
    clusters = cluster_entities(names, embeddings, threshold=args.threshold)

    multi_clusters = [c for c in clusters if len(c) > 1]
    singleton_count = sum(1 for c in clusters if len(c) == 1)
    print(f"Clusters: {len(clusters)} total ({len(multi_clusters)} multi-member, {singleton_count} singletons)")

    if not multi_clusters:
        print("No duplicates found at this threshold.")
        return

    if args.dry_run:
        print("\n[DRY RUN] Planned merges:")

    total_merged = 0
    for cluster in multi_clusters:
        canonical = pick_canonical(names, cluster)
        duplicates = [names[i] for i in cluster]
        print(f"\nCluster canonical='{canonical}':")
        for name in duplicates:
            if name != canonical:
                print(f"  - '{name}'")
        merge_cluster(driver, canonical, duplicates, dry_run=args.dry_run)
        total_merged += len(duplicates) - 1

    if args.dry_run:
        print(f"\n[DRY RUN] Would merge {total_merged} duplicate nodes.")
    else:
        print(f"\nDone. Merged {total_merged} duplicate Entity nodes.")
        print("Run check_layer3.py to verify graph health.")


if __name__ == "__main__":
    main()
