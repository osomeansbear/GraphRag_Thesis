import os
import re
import time
from dotenv import load_dotenv
from typing import ClassVar, List
from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore
from llama_index.core import Settings
from neo4j import GraphDatabase

load_dotenv()

# Model list for graph construction (quality first, high-capacity fallbacks).
CONSTRUCTION_MODEL_PRIORITY: List[str] = [
    "llama-3.3-70b-versatile",
    "qwen/qwen3-32b",
    "llama-3.1-8b-instant",
]


class FallbackGroqLLM(Groq):
    """
    Groq LLM wrapper that rotates through a model list on rate-limit errors.

    Strategies:
      "wait"       -- RPM limit: sleep and retry same model.
                      Daily limit: advance to next model permanently.
                      Use for evaluation runs where model consistency matters.
      "roundrobin" -- RPM limit: rotate to next model immediately.
                      Daily limit: remove model from rotation permanently.
                      Use for graph construction where speed matters.

    Pass a custom models list to override the default priority order.
    See CONSTRUCTION_MODEL_PRIORITY and GENERATION_MODEL_PRIORITY above.
    """

    MODEL_PRIORITY: ClassVar[List[str]] = CONSTRUCTION_MODEL_PRIORITY

    def __init__(self, strategy: str = "wait", models: List[str] = None):
        _models = models or self.MODEL_PRIORITY
        api_key = os.getenv("GROQ_API_KEY")
        super().__init__(model=_models[0], api_key=api_key, temperature=0.0, max_retries=0)
        self._models = _models
        self._instances = [
            Groq(model=m, api_key=api_key, temperature=0.0, max_retries=0)
            for m in _models
        ]
        self._strategy = strategy
        self._active_idx = 0
        self._rr_active = list(range(len(_models)))
        self._rr_ptr = 0

    @staticmethod
    def _is_rate_limited(exc: Exception) -> bool:
        msg = str(exc).lower()
        return any(x in msg for x in ["429", "rate limit", "quota", "tokens per", "exhausted"])

    @staticmethod
    def _is_rpm_limit(exc: Exception) -> bool:
        msg = str(exc).lower()
        return "requests per minute" in msg or "rpm" in msg

    @staticmethod
    def _retry_after(exc: Exception) -> float:
        m = re.search(r"try again in\s+([\d.]+)s", str(exc), re.IGNORECASE)
        return float(m.group(1)) + 0.5 if m else 5.0

    def _call_wait(self, method_name: str, *args, **kwargs):
        while self._active_idx < len(self._instances):
            i = self._active_idx
            try:
                return getattr(self._instances[i], method_name)(*args, **kwargs)
            except Exception as exc:
                if not self._is_rate_limited(exc):
                    raise
                if self._is_rpm_limit(exc):
                    wait = self._retry_after(exc)
                    print(f"[{self._models[i]}] RPM limit -- waiting {wait:.1f}s then retrying...")
                    time.sleep(wait)
                    continue
                if i + 1 < len(self._instances):
                    self._active_idx += 1
                    print(f"[{self._models[i]}] exhausted -> switching to [{self._models[self._active_idx]}]")
                    continue
                raise
        raise RuntimeError("All Groq models exhausted.")

    def _call_roundrobin(self, method_name: str, *args, **kwargs):
        consecutive_rpm = 0
        while self._rr_active:
            idx = self._rr_active[self._rr_ptr % len(self._rr_active)]
            model_name = self._models[idx]
            try:
                consecutive_rpm = 0
                return getattr(self._instances[idx], method_name)(*args, **kwargs)
            except Exception as exc:
                if not self._is_rate_limited(exc):
                    raise
                if self._is_rpm_limit(exc):
                    consecutive_rpm += 1
                    self._rr_ptr = (self._rr_ptr + 1) % len(self._rr_active)
                    if consecutive_rpm >= len(self._rr_active):
                        print(f"All {len(self._rr_active)} models RPM limited -- waiting 30s...")
                        time.sleep(30)
                        consecutive_rpm = 0
                    else:
                        next_model = self._models[self._rr_active[self._rr_ptr % len(self._rr_active)]]
                        print(f"[{model_name}] RPM limit -> rotating to [{next_model}]")
                else:
                    print(f"[{model_name}] daily limit -> removing from rotation")
                    self._rr_active.remove(idx)
                    if self._rr_active:
                        self._rr_ptr %= len(self._rr_active)
                    consecutive_rpm = 0
        raise RuntimeError("All Groq models exhausted from rotation.")

    def _call_with_fallback(self, method_name: str, *args, **kwargs):
        if self._strategy == "roundrobin":
            return self._call_roundrobin(method_name, *args, **kwargs)
        return self._call_wait(method_name, *args, **kwargs)

    def chat(self, messages, **kwargs):
        return self._call_with_fallback("chat", messages, **kwargs)

    def complete(self, prompt, **kwargs):
        return self._call_with_fallback("complete", prompt, **kwargs)

    def stream_chat(self, messages, **kwargs):
        return self._call_with_fallback("stream_chat", messages, **kwargs)

    def stream_complete(self, prompt, **kwargs):
        return self._call_with_fallback("stream_complete", prompt, **kwargs)


def _build_llm():
    """
    Build the graph-construction LLM (FallbackGroqLLM with model rotation).

      LLM_STRATEGY=wait        for evaluation / thesis runs (accuracy, default)
      LLM_STRATEGY=roundrobin  for live demo (speed)

    Note: generation and reflection use SingleGroqLLM (pinned llama-3.1-8b-instant),
    instantiated directly in run_3x3.py and retriever.py, not here.
    """
    strategy = os.getenv("LLM_STRATEGY", "wait")
    print(f"[Config] LLM: Groq (strategy={strategy})")
    return FallbackGroqLLM(strategy=strategy)


class SettingsConfig:
    """Global config. Graph-construction LLM uses FallbackGroqLLM (see _build_llm)."""

    llm = _build_llm()

    embed_model = HuggingFaceEmbedding(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        device="cpu"
    )

    neo4j_driver = GraphDatabase.driver(
        os.getenv("NEO4J_URI"),
        auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD"))
    )

    @staticmethod
    def get_graph_store():
        return Neo4jPropertyGraphStore(
            username=os.getenv("NEO4J_USERNAME"),
            password=os.getenv("NEO4J_PASSWORD"),
            url=os.getenv("NEO4J_URI"),
        )

    @classmethod
    def setup_global_settings(cls):
        Settings.llm = cls.llm
        Settings.embed_model = cls.embed_model
        Settings.chunk_size = 512
        Settings.chunk_overlap = 50

SettingsConfig.setup_global_settings()
