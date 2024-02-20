from pathlib import Path
from llama_index.readers.file import PDFReader
from llama_index.readers.file import PyMuPDFReader
from llama_index.core import Document
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter, HierarchicalNodeParser, get_leaf_nodes, get_root_nodes
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.vector_stores import SimpleVectorStore
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.llms.openai import OpenAI
from llama_index.core.retrievers import AutoMergingRetriever
from dotenv import load_dotenv
import os

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
loader = PyMuPDFReader()
# by default, create documents by page
document = loader.load_data(file_path=Path("./data/llama2.pdf"))
# merge all the text to rebuild node
doc_text = "\n\n".join([d.get_content() for d in document])
documents = [Document(text=doc_text)]

# HierarchicalNodeParser will output different level of chunks : 2048, 512, 138
node_parser = HierarchicalNodeParser.from_defaults()
nodes = node_parser.get_nodes_from_documents(documents)
leaf_nodes = get_leaf_nodes(nodes)
root_nodes = get_root_nodes(nodes)

# store embedding into Chroma and create index

docstore = SimpleDocumentStore()
vector_store = SimpleVectorStore()
docstore.add_documents(nodes)
storage_context  = StorageContext.from_defaults(docstore=docstore, vector_store=vector_store)
base_index = VectorStoreIndex(leaf_nodes, storage_context=storage_context)

# query
llm = OpenAI(model="gpt-3.5-turbo")
base_retriever = base_index.as_retriever(similarity_top_k=6)
retriever = AutoMergingRetriever(base_retriever, storage_context, verbose=True)
# Using Auto merging
# can see from the output that some leaf nodes are replaced into parent nodes
# before being passed into LLM
nodes = retriever.retrieve("What could be the potential outcomes of adjusting the amount of safety"
    " data used in the RLHF stage?")
# Using 138 chunk size data
base_nodes = base_retriever.retrieve("What could be the potential outcomes of adjusting the amount of safety"
    " data used in the RLHF stage?")

query_engine = RetrieverQueryEngine.from_args(retriever=retriever)
base_query_engine = RetrieverQueryEngine.from_args(base_retriever)

response = query_engine.query("What could be the potential outcomes of adjusting the amount of safety"
    " data used in the RLHF stage?")
base_response = base_query_engine.query("What could be the potential outcomes of adjusting the amount of safety"
    " data used in the RLHF stage?")

print("\nAuto merging:\n", str(response))
print("\nBase:\n", str(base_response))








