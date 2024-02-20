from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.indices.query.query_transform.base import  HyDEQueryTransform
from llama_index.core.query_engine import TransformQueryEngine
from dotenv import load_dotenv
import os

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

documents = SimpleDirectoryReader(input_files=["./data/example1/paul_graham_essay.txt"]).load_data()
# if just build index by VectorStroeIndex(documents), getting error message that exceed max input token in embedding phase
index = VectorStoreIndex.from_documents(documents)

query_engine = index.as_query_engine()
query_str = "what did paul graham do after going to RISD"
response = query_engine.query(query_str)
print("\nBase response :\n",response)

hyde = HyDEQueryTransform(include_original=True)
hyde_query_engine = TransformQueryEngine(query_engine, hyde)
response = hyde_query_engine.query(query_str)
print("\nHyDE response :\n",response)

# query_str is transform into hypo_doc as new query for retrieval
hypo_doc = hyde(query_str)
print("\nHypothetical Answer : \n",hypo_doc.embedding_strs[0])

"""summary
1. if initial query is brief, it may be more appropriate to 
use initial query instead of new query(by HyDE) to do retrieval

2. for the query that is an open-ended question, 
the hypothetical answer may have negative effect in retrieval
"""
