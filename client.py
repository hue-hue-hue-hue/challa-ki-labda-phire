from pathway.xpacks.llm.vector_store import VectorStoreClient

PATHWAY_PORT = 8765
PATHWAY_HOST = "127.0.0.1"

client = VectorStoreClient(
    host=PATHWAY_HOST,
    port=PATHWAY_PORT,
)

ans = client.query("pathway?")
print(ans)