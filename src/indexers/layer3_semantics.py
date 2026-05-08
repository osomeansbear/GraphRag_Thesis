import os
import time
from typing import List, Literal
from pydantic import BaseModel, Field
import instructor
from openai import OpenAI
from src.core.config import SettingsConfig
from tqdm import tqdm

# --- 1. SCHEMA ---
class Entity(BaseModel):
    name: str = Field(..., description="Entity name (e.g., Student, GPA, Academic Warning)")
    label: str = Field(..., description="Type: 'Role', 'Object', 'Metric', 'Event'")

# Valid relation types — must match retriever's CAUSAL_PATTERNS / CONDITIONAL_PATTERNS
# Tier mapping (used for path scoring in MULTIHOP):
#   causal (weight 3):      REQUIRES, LEADS_TO
#   conditional (weight 2): AFFECTS, HAS_CONDITION, EXEMPTS
#   default (weight 1):     PERFORMS
RelationType = Literal[
    "REQUIRES",       # A cannot happen without B (hard prerequisite)     → causal
    "LEADS_TO",       # A directly causes B to happen                     → causal
    "AFFECTS",        # A influences the value or outcome of B            → conditional
    "HAS_CONDITION",  # A only applies when B is true                     → conditional
    "EXEMPTS",        # A is an exception that waives or overrides B      → conditional
    "PERFORMS",       # a person/role performs an action                  → default
    "NONE",           # no clear directional relationship exists          → skipped
]

VALID_RELATION_TYPES = {
    "REQUIRES", "LEADS_TO", "AFFECTS", "HAS_CONDITION", "EXEMPTS", "PERFORMS"
}


def _normalize_entity_name(name: str) -> str:
    """Strip whitespace and collapse internal spaces, preserving original casing.
    .title() is intentionally avoided — it mangles CamelCase and acronyms."""
    return " ".join(name.strip().split())

class Relationship(BaseModel):
    source: str = Field(..., description="Source entity name")
    target: str = Field(..., description="Target entity name")
    relation_type: RelationType = Field(
        ...,
        description=(
            "Relationship type. MUST be exactly one of: "
            "REQUIRES, LEADS_TO, AFFECTS, HAS_CONDITION, EXEMPTS, PERFORMS, NONE.\n"
            "  REQUIRES     — A cannot happen without B (e.g., Graduation REQUIRES MinimumGPA)\n"
            "  LEADS_TO     — A directly causes B (e.g., LowGPA LEADS_TO AcademicWarning)\n"
            "  AFFECTS      — A influences the value of B (e.g., CGPA AFFECTS GraduationRank)\n"
            "  HAS_CONDITION — A only applies when B holds (e.g., Scholarship HAS_CONDITION NoViolation)\n"
            "  EXEMPTS      — A waives or overrides B (e.g., InternationalCert EXEMPTS EnglishRequirement)\n"
            "  PERFORMS     — a role/actor performs an action (e.g., Student PERFORMS Registration)\n"
            "  NONE         — no clear directional relationship; use only when none above fits."
        )
    )
    evidence: str = Field(
        ...,
        description="Exact short quote from the source text that supports this relationship."
    )

class ExtractionResult(BaseModel):
    entities: List[Entity]
    relationships: List[Relationship]

# --- 2. EXTRACTION FUNCTION ---
# Models verified against the current Groq lineup (2026-04). Order = quality-first
# with fallbacks chosen for high TPM / RPM headroom and instructor-tools compatibility.
# Scout is included here (graph build only) — RAGAS uses llama-3.3-70b directly.
MODELS_FALLBACK = [
    "llama-3.3-70b-versatile",                    # primary (strongest, but TPM=12K is tight)
    "meta-llama/llama-4-scout-17b-16e-instruct",  # TPM=30K — best fallback for throughput
    "qwen/qwen3-32b",                             # RPM=60 (2x), strong reasoning
    "llama-3.1-8b-instant",                       # last-resort fast/cheap
]

FEW_SHOT_EXAMPLES = """
EXAMPLE 1
Text: "A student whose cumulative GPA falls below 2.0 at the end of any semester will receive an academic warning."
Entities: Student, CumulativeGPA, AcademicWarning
Relationships:
  - CumulativeGPA LEADS_TO AcademicWarning | evidence: "cumulative GPA falls below 2.0 ... will receive an academic warning"
  - Student PERFORMS AcademicProgress | evidence: "A student whose cumulative GPA"

EXAMPLE 2
Text: "To graduate with distinction, a student must have a CGPA of at least 3.60 and must not have received any disciplinary action."
Entities: Student, CGPA, GraduationWithDistinction, DisciplinaryAction
Relationships:
  - GraduationWithDistinction REQUIRES CGPA | evidence: "must have a CGPA of at least 3.60"
  - GraduationWithDistinction HAS_CONDITION NoDisciplinaryAction | evidence: "must not have received any disciplinary action"

EXAMPLE 3
Text: "Students who hold a valid international English certificate are exempt from the university English proficiency requirement."
Entities: Student, InternationalEnglishCertificate, EnglishProficiencyRequirement
Relationships:
  - InternationalEnglishCertificate EXEMPTS EnglishProficiencyRequirement | evidence: "are exempt from the university English proficiency requirement"
"""

SYSTEM_PROMPT = (
    "You are an academic regulation analyst. "
    "Extract entities and relationships from regulation clause text. "
    "Relationship type MUST be exactly one of: "
    "REQUIRES, LEADS_TO, AFFECTS, HAS_CONDITION, EXEMPTS, PERFORMS, NONE. "
    "Use NONE only when no other type fits. "
    "Always include an evidence field with a short exact quote from the text."
)


def extract_semantics_manual(text: str, client, model: str = "llama-3.3-70b-versatile"):
    if len(text) > 3000:
        text = text[:3000] + "...(truncated)"

    user_prompt = f"{FEW_SHOT_EXAMPLES}\nNow extract from:\nText: \"{text}\""

    return client.chat.completions.create(
        model=model,
        response_model=ExtractionResult,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_prompt},
        ],
        temperature=0.0,
    )

def extract_with_fallback(text: str, client) -> tuple[ExtractionResult, str]:
    """
    Try each model in MODELS_FALLBACK order; rotate on transient errors:
      429 (rate-limit), 404 (model deprecated), 400 (model-specific bad request),
      500/502/503/504 (server-side). Anything else fails immediately.
    Returns (result, model_name_used) so callers can record `extracted_by`
    on the resulting relations for thesis Appendix B disclosure.
    """
    ROTATABLE_CODES = ("429", "404", "400", "500", "502", "503", "504")
    for i, model in enumerate(MODELS_FALLBACK):
        try:
            result = extract_semantics_manual(text, client, model=model)
            return result, model
        except Exception as exc:
            is_last = i + 1 == len(MODELS_FALLBACK)
            err = str(exc)
            matched = next((c for c in ROTATABLE_CODES if c in err), None)
            if matched and not is_last:
                print(f"\n[{model}] HTTP {matched} -> switching to [{MODELS_FALLBACK[i + 1]}]")
                continue
            raise

# --- 3. MAIN PIPELINE V28 ---
def build_layer_23_semantics_v28():
    print("--- LAYER 2 & 3 (V28): SEMANTIC EXTRACTION ---")
    store = SettingsConfig.get_graph_store()

    # Setup Instructor client (single client, model is passed per-call)
    groq_key = os.getenv("GROQ_API_KEY")
    client = instructor.from_openai(
        OpenAI(base_url="https://api.groq.com/openai/v1", api_key=groq_key),
        mode=instructor.Mode.TOOLS
    )

    # Load data
    # Single query: only clauses with content that have no MENTIONS edges yet
    print("Finding unprocessed clauses...")
    results = store.client.execute_query("""
        MATCH (c:Clause)
        WHERE c.content_raw IS NOT NULL
          AND NOT (c)-[:MENTIONS]->()
        RETURN c.id AS id, c.content_raw AS text
    """)
    pending_clauses = [{"id": r["id"], "text": r["text"]} for r in results.records]

    print(f"Pending: {len(pending_clauses)} clauses. Starting extraction...")

    success_count = 0
    fail_count = 0

    for item in tqdm(pending_clauses, desc="Semantic Extraction"):
        clause_id = item["id"]
        text = item["text"]

        if len(text) < 15:
            continue

        time.sleep(5)

        try:
            data, model_used = extract_with_fallback(text, client)

            for ent in data.entities:
                norm_name = _normalize_entity_name(ent.name)
                store.client.execute_query("""
                    MATCH (c:Clause {id: $cid})
                    MERGE (e:Entity {name: $name})
                    ON CREATE SET e.label = $label, e.extracted_by = $model_used
                    MERGE (c)-[m:MENTIONS]->(e)
                    ON CREATE SET m.extracted_by = $model_used
                """, cid=clause_id, name=norm_name, label=ent.label, model_used=model_used)

            for rel in data.relationships:
                # Safety filter: skip NONE and any type outside valid schema
                if rel.relation_type not in VALID_RELATION_TYPES:
                    continue
                rel_type = rel.relation_type
                src_norm = _normalize_entity_name(rel.source)
                tgt_norm = _normalize_entity_name(rel.target)
                # MERGE (not MATCH) so mismatched casing from LLM never silently drops the edge
                query = f"""
                    MERGE (a:Entity {{name: $src}})
                    MERGE (b:Entity {{name: $tgt}})
                    MERGE (a)-[r:`{rel_type}`]->(b)
                    ON CREATE SET r.evidence = $evidence, r.extracted_by = $model_used
                """
                store.client.execute_query(
                    query,
                    src=src_norm, tgt=tgt_norm,
                    evidence=rel.evidence, model_used=model_used,
                )

            success_count += 1

        except Exception as exc:
            print(f"\nSkipping {clause_id}: {str(exc)[:80]}...")
            fail_count += 1
            continue

    print(f"\nDONE! Success: {success_count}, Failed: {fail_count}")
    print("Run Graph Health Check to review results.")

if __name__ == "__main__":
    build_layer_23_semantics_v28()
