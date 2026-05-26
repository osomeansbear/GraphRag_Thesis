import os
from dotenv import load_dotenv
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore
from llama_index.core import Settings
from neo4j import GraphDatabase

load_dotenv()


class SettingsConfig:

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
        Settings.embed_model = cls.embed_model
        Settings.chunk_size = 512
        Settings.chunk_overlap = 50

SettingsConfig.setup_global_settings()
