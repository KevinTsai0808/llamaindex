from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.query_engine import RetrieverQueryEngine
from dotenv import load_dotenv
import os

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

documents = SimpleDirectoryReader("data/example1").load_data()
text_splitter = SentenceSplitter(chunk_size=300)

index = VectorStoreIndex.from_documents(documents, transformations=[text_splitter])
# print(index.docstore)

# first retriever
vector_retriever = index.as_retriever(similarity_top_k=2)
# second retriever
bm25_retriever = BM25Retriever.from_defaults(
    docstore=index.docstore, similarity_top_k=2
)

# fusion both retriever using Reciprocal Rerank Fusion(RRF)
retriever = QueryFusionRetriever(
    [vector_retriever, bm25_retriever],
    similarity_top_k=2,
    num_queries=4,
    mode="reciprocal_rerank",
    verbose=True,
    # default query_gen_prompt="
    # You are a helpful assistant that generates multiple search queries based on a 
    # single input query. Generate {num_queries} search queries, one on each line, 
    # related to the following input query:\n
    # Query: {query}\n
    # Queries:\n"
)

query_engine = RetrieverQueryEngine.from_args(retriever)
response = query_engine.query("What happened at Interleafe and Viaweb?")
print("\nResponse : \n", response)
