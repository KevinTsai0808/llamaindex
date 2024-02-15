from llama_index import SimpleDirectoryReader, VectorStoreIndex
# basic
documents = SimpleDirectoryReader("./data").load_data
index = VectorStoreIndex.from_documents(documents)
query_engine  = index.as_query_engine()
response = query_engine.query(
    "Write an email to the user given their background information."
)
print(response)

"""
ther are three main steps to do in querying:
1. Retrieval : most common is the top-k retrieval
2. Postprocessing : such as reranked, filtering, or transformation
3. Response synthesis : combine the retrieved documents and prompt and send to LLM
"""

from llama_index.core import VectorStoreIndex, get_response_synthesizer,  VectorIndexRetriever, RetrieverQueryEngine, SimilarityPostprocessor
# build index
index = VectorStoreIndex.from_documents(documents)
# configure retriever
retriever = VectorIndexRetriever(
    index=index,
    similarity_top_k=10,
)
# configure response synthesizer
response_synthesizer = get_response_synthesizer()
# assemble query engine
query_engine = RetrieverQueryEngine(
    retriever=retriever,
    response_synthesizer=response_synthesizer,
    node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.7)],
)
# query
response = query_engine.query("What did the author do growing up?")
print(response)
