import re
import unicodedata
from docling.document_converter import DocumentConverter
from src.core.config import SettingsConfig

def build_layer_1_structure():

    store = SettingsConfig.get_graph_store()
    store.client.execute_query("MATCH (n) DETACH DELETE n")

    print("Reading PDF...")
    converter = DocumentConverter()
    result = converter.convert("data/raw/quy_dinh_dao_tao_EN.pdf")
    raw_text = result.document.export_to_markdown()

    # 1. Normalize Unicode
    text = unicodedata.normalize('NFC', raw_text)

    # 2. Basic formatting fixes
    # Split Chapter/Article lines that are merged
    text = re.sub(r'([A-Z]+)\s+(Article \d+)', r'\1\n\2', text)
    # Split point markers (a, b, c)
    text = re.sub(r'(\s)([a-zA-Z][\)\.])', r'\n\2', text)

    all_lines = [line.strip() for line in text.split('\n') if line.strip()]

    # --- STEP 1: FIND START OF MAIN CONTENT ---
    print("Locating document start...")
    start_index = -1
    for i, line in enumerate(all_lines):
        if "GENERAL PROVISIONS" in line.upper():
            start_index = i
            break

    if start_index != -1:
        print(f"   Trimmed {start_index} header lines.")
        processing_lines = all_lines[start_index:]
        processing_lines[0] = "Chapter I GENERAL PROVISIONS"
    else:
        print("Warning: 'GENERAL PROVISIONS' not found. Using full file.")
        processing_lines = all_lines

    # Init root Document node
    doc_id = "DOC_719_EN"
    store.client.execute_query("""
        CREATE (d:Document {
            id: $id, name: 'IU Training Regulation EN',
            structure: 'V24 List Ops'
        })
    """, id=doc_id)

    # --- REGEX PATTERNS ---
    REGEX_CHAP = r'.*Chapter\s+([IVX0-9]+)'
    REGEX_ART  = r'^\W*Article\s+(\d+)'
    REGEX_CLAUSE = r'^\W*(\d+)\s*[\.\)]'
    REGEX_POINT_DETECT = r'^[^\w\n]*([a-zA-Z])\s*[\)\.]'

    curr_chap_id, curr_art_id, curr_clause_id = None, None, None
    buffer = []
    last_node_id, last_label = None, None

    global_art_num = 0
    current_clause_num = 0

    skip_next = False

    def flush():
        nonlocal buffer, last_node_id, last_label
        if last_node_id and buffer:
            text = "\n".join(buffer).strip()
            if text:
                store.client.execute_query(f"""
                    MATCH (n:{last_label} {{id: $id}}) SET n.content_raw = $text
                """, id=last_node_id, text=text)
            buffer = []

    print(f"Scanning {len(processing_lines)} lines...")

    for i, line in enumerate(processing_lines):
        if skip_next:
            skip_next = False
            continue

        # 1. CHAPTER (with look-ahead for multi-line titles)
        if re.match(REGEX_CHAP, line, re.IGNORECASE):
            flush()
            match = re.match(REGEX_CHAP, line, re.IGNORECASE)

            current_title = re.sub(r'[#\*]', '', line).strip()
            final_title = current_title

            if i + 1 < len(processing_lines):
                next_line = processing_lines[i+1].strip()
                is_upper = next_line.isupper()
                is_not_art = not re.match(REGEX_ART, next_line)
                is_not_chap = not re.match(REGEX_CHAP, next_line)

                if is_upper and is_not_art and is_not_chap:
                    print(f"   Merging title: '{current_title}' + '{next_line}'")
                    final_title = f"{current_title}: {re.sub(r'[#\*]', '', next_line)}"
                    skip_next = True

            curr_chap_id = f"chap_{i}"
            print(f"[L{i}] {final_title}")
            store.client.execute_query("""
                MATCH (d:Document {id: $did}) CREATE (c:Chapter {id: $id, title: $t}) CREATE (d)-[:HAS_CHAPTER]->(c)
            """, did=doc_id, id=curr_chap_id, t=final_title)

            curr_art_id, curr_clause_id = None, None
            current_clause_num = 0

        # 2. ARTICLE
        elif re.match(REGEX_ART, line, re.IGNORECASE):
            match = re.search(REGEX_ART, line, re.IGNORECASE)
            art_num = int(match.group(1))

            if art_num <= global_art_num:
                if last_node_id: buffer.append(line)
                continue

            flush()
            if curr_chap_id is None:
                curr_chap_id = "chap_1_auto"
                store.client.execute_query("MATCH (d:Document {id:$did}) MERGE (c:Chapter {id:$id}) MERGE (d)-[:HAS_CHAPTER]->(c)", did=doc_id, id=curr_chap_id)

            global_art_num = art_num
            title = re.sub(r'[#\*]', '', line).strip()
            curr_art_id = f"art_{i}"
            print(f"  [L{i}] Article {art_num}: {title}")
            store.client.execute_query("""
                MATCH (c:Chapter {id: $cid}) CREATE (a:Article {id: $id, title: $t, art_num: $num}) CREATE (c)-[:HAS_ARTICLE]->(a)
            """, cid=curr_chap_id, id=curr_art_id, t=title, num=art_num)

            curr_clause_id = None
            current_clause_num = 0

        # 3. CLAUSE
        elif re.match(REGEX_CLAUSE, line) and curr_art_id:
            match = re.match(REGEX_CLAUSE, line)
            new_clause_num = int(match.group(1))

            is_start = (new_clause_num == 1 and current_clause_num == 0)
            is_sequential = (new_clause_num == current_clause_num + 1)
            is_small_jump = (new_clause_num > current_clause_num) and (new_clause_num <= current_clause_num + 3)

            if is_start or is_sequential or is_small_jump:
                flush()
                curr_clause_id = f"clause_{i}"
                last_node_id, last_label = curr_clause_id, "Clause"
                current_clause_num = new_clause_num

                print(f"    [L{i}] Clause {new_clause_num}")
                store.client.execute_query("""
                    MATCH (a:Article {id: $aid}) CREATE (cl:Clause {id: $id, clause_num: $num}) CREATE (a)-[:HAS_CLAUSE]->(cl)
                """, aid=curr_art_id, id=curr_clause_id, num=new_clause_num)
                buffer.append(line)
            else:
                if last_node_id: buffer.append(line)

        # 4. POINT (sub-clause like a, b, c)
        elif re.match(REGEX_POINT_DETECT, line):
            if curr_clause_id is None and curr_art_id:
                curr_clause_id = f"clause_{i}_auto"
                last_node_id, last_label = curr_clause_id, "Clause"
                current_clause_num = 1
                store.client.execute_query("""
                    MATCH (a:Article {id: $aid}) CREATE (cl:Clause {id: $id, clause_num: $num, is_auto: true}) CREATE (a)-[:HAS_CLAUSE]->(cl)
                """, aid=curr_art_id, id=curr_clause_id, num=current_clause_num)
            buffer.append(line)

        # 5. PLAIN TEXT
        else:
            if last_node_id: buffer.append(line)

    flush()
    print("All titles processed and graph nodes created.")

if __name__ == "__main__":
    build_layer_1_structure()
