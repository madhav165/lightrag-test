"""
LightRAG Demo Script

This script demonstrates the capabilities of the LightRAG framework by
initializing a RAG instance, inserting sample text data, and performing
various query modes (naive, local, global, hybrid). It also includes
robust logging configuration and error handling.
"""

import asyncio
import logging
import logging.config
import os
import random

import networkx as nx
import numpy as np
from lightrag import LightRAG, QueryParam
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.utils import EmbeddingFunc, logger, set_verbose_debug
from pyvis.network import Network

WORKING_DIR = "./demo"


def configure_logging():
    """Configure logging for the application"""

    # Reset any existing handlers to ensure clean configuration
    for logger_name in ["uvicorn", "uvicorn.access", "uvicorn.error", "lightrag"]:
        logger_instance = logging.getLogger(logger_name)
        logger_instance.handlers = []
        logger_instance.filters = []

    # Get log directory path from environment variable or use current directory
    log_dir = os.getenv("LOG_DIR", os.getcwd())
    log_file_path = os.path.abspath(os.path.join(log_dir, "lightrag_demo.log"))

    print(f"\nLightRAG demo log file: {log_file_path}\n")
    os.makedirs(os.path.dirname(log_dir), exist_ok=True)

    # Get log file max size and backup count from environment variables
    log_max_bytes = int(os.getenv("LOG_MAX_BYTES", "10485760"))  # Default 10MB
    log_backup_count = int(os.getenv("LOG_BACKUP_COUNT", "5"))  # Default 5 backups

    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {
                    "format": "%(levelname)s: %(message)s",
                },
                "detailed": {
                    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                },
            },
            "handlers": {
                "console": {
                    "formatter": "default",
                    "class": "logging.StreamHandler",
                    "stream": "ext://sys.stderr",
                },
                "file": {
                    "formatter": "detailed",
                    "class": "logging.handlers.RotatingFileHandler",
                    "filename": log_file_path,
                    "maxBytes": log_max_bytes,
                    "backupCount": log_backup_count,
                    "encoding": "utf-8",
                },
            },
            "loggers": {
                "lightrag": {
                    "handlers": ["console", "file"],
                    "level": "INFO",
                    "propagate": False,
                },
            },
        }
    )

    # Set the logger level to INFO
    logger.setLevel(logging.INFO)
    # Enable verbose debug if needed
    set_verbose_debug(os.getenv("VERBOSE_DEBUG", "false").lower() == "true")


if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)


async def llm_model_func(
    prompt, system_prompt=None, history_messages=None, **kwargs
) -> str:
    """Function to call OpenAI LLM model for text completion"""
    return await openai_complete_if_cache(
        "gpt-5-nano",
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url="https://api.openai.com/v1",
        **kwargs,
    )


async def embedding_func(texts: list[str]) -> np.ndarray:
    """Generate embeddings for a list of texts using OpenAI embedding model"""
    return await openai_embed(
        texts,
        model="text-embedding-3-small",
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url="https://api.openai.com/v1",
    )


async def initialize_rag():
    """Initialize and return a LightRAG instance"""
    rag = LightRAG(
        working_dir=WORKING_DIR,
        embedding_func=EmbeddingFunc(embedding_dim=1536, func=embedding_func),
        llm_model_func=llm_model_func,
    )

    await rag.initialize_storages()
    await initialize_pipeline_status()

    return rag


def delete_old_files(files: list[str]):
    """Delete old data files from the working directory"""
    for file in files:
        file_path = os.path.join(WORKING_DIR, file)
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"Deleting old file:: {file_path}")


def visualize_graph():
    """Visualize the knowledge graph using Pyvis"""
    graph = nx.read_graphml("./demo/graph_chunk_entity_relation.graphml")

    # Create a Pyvis network
    net = Network(height="100vh", notebook=True)

    # Convert NetworkX graph to Pyvis network
    net.from_nx(graph)

    # Add colors and title to nodes
    for node in net.nodes:
        hex_value = random.randint(0, 0xFFFFFF)
        hex_color = f"#{hex_value:06x}"
        node["color"] = hex_color
        if "description" in node:
            node["title"] = node["description"]

    # Add title to edges
    for edge in net.edges:
        if "description" in edge:
            edge["title"] = edge["description"]

    # Save and display the network
    net.show("./demo/knowledge_graph.html")


async def main():
    """Main function to demonstrate LightRAG capabilities"""
    if not os.getenv("OPENAI_API_KEY"):
        print(
            "Error: OPENAI_API_KEY environment variable is not set. "
            "Please set this variable before running the program."
        )
        print("You can set the environment variable by running:")
        print("  export OPENAI_API_KEY='your-openai-api-key'")
        return  # Exit the async function

    try:
        # Clear old data files
        files_to_delete = [
            "graph_chunk_entity_relation.graphml",
            "kv_store_doc_status.json",
            "kv_store_full_docs.json",
            "kv_store_text_chunks.json",
            "vdb_chunks.json",
            "vdb_entities.json",
            "vdb_relationships.json",
            "knowledge_graph.html",
        ]

        delete_old_files(files_to_delete)

        # Initialize RAG instance
        rag = await initialize_rag()

        with open("./sample2.txt", "r", encoding="utf-8") as f:
            await rag.ainsert(f.read())

        visualize_graph()

        # "local_mode": "Focuses on specific entities and their relationships",
        # "global_mode": "Provides broader context from relationship patterns",
        # "hybrid_mode": "Combines local and global approaches",
        # "naive_mode": "Simple vector similarity search",
        # "mix_mode": "Integrates knowledge graph and vector retrieval",

        # Perform naive search
        print("\n=====================")
        print("Query mode: naive")
        print("=====================")
        print(
            await rag.aquery(
                "Are we all in the same time?",
                param=QueryParam(
                    mode="naive",
                    enable_rerank=False,
                    top_k=10,
                    chunk_top_k=5,
                ),
            )
        )

        # Perform local search
        print("\n=====================")
        print("Query mode: local")
        print("=====================")
        print(
            await rag.aquery(
                "Are we all in the same time?",
                param=QueryParam(
                    mode="local",
                    enable_rerank=False,
                    top_k=10,
                    chunk_top_k=5,
                ),
            )
        )

        # Perform global search
        print("\n=====================")
        print("Query mode: global")
        print("=====================")
        print(
            await rag.aquery(
                "Are we all in the same time?",
                param=QueryParam(
                    mode="global",
                    enable_rerank=False,
                    top_k=10,
                    chunk_top_k=5,
                ),
            )
        )

        # Perform hybrid search
        print("\n=====================")
        print("Query mode: hybrid")
        print("=====================")
        print(
            await rag.aquery(
                "Are we all in the same time?",
                param=QueryParam(
                    mode="hybrid",
                    enable_rerank=False,
                    top_k=10,
                    chunk_top_k=5,
                ),
            )
        )

        # Perform mix search
        print("\n=====================")
        print("Query mode: mix")
        print("=====================")
        print(
            await rag.aquery(
                "Are we all in the same time?",
                param=QueryParam(
                    mode="mix",
                    enable_rerank=False,
                    top_k=10,
                    chunk_top_k=5,
                ),
            )
        )
    except Exception as e:  # pylint: disable=broad-exception-caught
        if isinstance(e, KeyboardInterrupt):
            print("\nProcess interrupted by user.")
        if isinstance(e, ConnectionError):
            print("\nConnection error occurred.")
        if isinstance(e, RuntimeError):
            print(f"Runtime error: {e}")
        else:
            print(f"An error occurred: {e}")
            raise
    finally:
        if rag:
            await rag.finalize_storages()


if __name__ == "__main__":
    # Configure logging before running the main function
    configure_logging()
    asyncio.run(main())
    print("\nDone!")
