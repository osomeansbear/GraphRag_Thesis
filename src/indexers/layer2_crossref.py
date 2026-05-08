import re
from src.core.config import SettingsConfig

def build_layer_1_crossref_v30_precision():
    print("--- LAYER 1.5 (V30): PRECISION CROSS-REFERENCE ---")

    store = SettingsConfig.get_graph_store()

    # --- STEP 1: BUILD ADDRESS BOOK ---
    print("Building detailed address map (Articles & Clauses)...")

    # 1.1 Map Article: art_num -> Node ID (uses stored art_num property)
    art_map = {}
    res_art = store.client.execute_query(
        "MATCH (a:Article) RETURN a.id as id, a.art_num as num"
    )[0]
    for r in res_art:
        if r['num'] is not None:
            art_map[str(r['num'])] = r['id']
        else:
            # Fallback: re-parse title for nodes built before art_num was stored
            m = re.search(r'Article\s+0*(\d+)', r.get('title', ''), re.IGNORECASE)
            if m: art_map[m.group(1)] = r['id']

    # 1.2 Map Clause: (art_num_str, clause_num_str) -> Node ID (uses stored properties)
    clause_map = {}
    res_clause = store.client.execute_query("""
        MATCH (a:Article)-[:HAS_CLAUSE]->(c:Clause)
        RETURN a.art_num as art_num, c.clause_num as clause_num, c.id as id
    """)[0]
    for r in res_clause:
        if r['art_num'] is not None and r['clause_num'] is not None:
            clause_map[(str(r['art_num']), str(r['clause_num']))] = r['id']

    print(f"Address map: {len(art_map)} Articles, {len(clause_map)} Clauses.")

    # --- STEP 2: SCAN AND CREATE LINKS ---
    content_query = """
        MATCH (c:Clause)<-[:HAS_CLAUSE]-(parent:Article)
        WHERE c.content_raw IS NOT NULL
        RETURN c.id as id, c.content_raw as text, parent.id as parent_id
    """
    nodes = store.client.execute_query(content_query)[0]

    count_deep = 0
    count_wide = 0

    # Regex for: "Clause 10 ... Article 2"
    REGEX_DEEP = r'[Cc]lause\s+(\d+)(?:[^.]{0,30})[Aa]rticle\s+(\d+)'

    # Regex for standalone: "Article 2"
    REGEX_WIDE = r'[Aa]rticle\s+(\d+)'

    for node in nodes:
        raw_text = node['text']
        clean_text = re.sub(r'\s+', ' ', raw_text)

        source_id = node['id']
        parent_id = node['parent_id']

        # --- A. DEEP LINKS (Clause X Article Y) ---
        deep_matches = re.finditer(REGEX_DEEP, clean_text)
        found_ranges = []

        for m in deep_matches:
            c_num = m.group(1)
            a_num = m.group(2)

            found_ranges.append(m.span())

            key = (a_num, c_num)
            if key in clause_map:
                target_id = clause_map[key]
                if target_id != source_id:
                    store.client.execute_query("""
                        MATCH (s:Clause {id: $sid}), (t:Clause {id: $tid})
                        MERGE (s)-[r:REFERS_TO]->(t)
                        SET r.type = 'deep_link', r.desc = 'Specific reference: Clause ' + $c + ' Article ' + $a
                    """, sid=source_id, tid=target_id, c=c_num, a=a_num)
                    count_deep += 1
                    print(f"   [Deep] {source_id} -> Clause {c_num} Article {a_num}")

        # --- B. WIDE LINKS (Article Y only) ---
        # Only match "Article Y" not already captured as part of a deep link
        wide_matches = re.finditer(REGEX_WIDE, clean_text)

        for m in wide_matches:
            start, end = m.span()
            is_overlap = any(start >= fs and end <= fe for (fs, fe) in found_ranges)
            if is_overlap:
                continue

            a_num = m.group(1)
            if a_num in art_map:
                target_id = art_map[a_num]
                if target_id != parent_id:
                    store.client.execute_query("""
                        MATCH (s:Clause {id: $sid}), (t:Article {id: $tid})
                        MERGE (s)-[r:REFERS_TO]->(t)
                        SET r.type = 'article_link'
                    """, sid=source_id, tid=target_id)
                    count_wide += 1

    print(f"V30 complete: {count_deep} deep links (clause-level), {count_wide} article links.")

if __name__ == "__main__":
    build_layer_1_crossref_v30_precision()
