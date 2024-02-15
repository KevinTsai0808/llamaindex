import os
import chromadb
from dotenv import load_dotenv
from llama_index.llms import PaLM
from llama_index.vector_stores import ChromaVectorStore
from llama_index import VectorStoreIndex,SimpleDirectoryReader,ServiceContext,StorageContext
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Adjust some params like chunk size, LLM
service_context = ServiceContext.from_defaults(chunk_size=1000)


# if want to change different vector store
# initialize client, setting path to save data
chroma_client = chromadb.PersistentClient(path="./testing_chroma")
# create or loading collection
# chroma_collection = chroma_client.create_collection("testing")
chroma_collection = chroma_client.get_or_create_collection("testing")
# assign chroma as the vector_store to the context
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context  = StorageContext.from_defaults(vector_store=vector_store)

# Strp1:Loading
document = SimpleDirectoryReader("data").load_data()

# Step2:Indexing and Storing
index = VectorStoreIndex.from_documents(document, service_context=service_context, storage_context=storage_context)

# Step3:Querying
response_mode = "refine"
"""
refine : take first chunk to LLM for first response(text_qa_template), then the response will be concated with chunk2 for response2(refinr_template),.....
compact: repack + refine
tree_summarize : top k chunks list -> repack -> LLM -> response_list -> repack -> LLM .....
simple_summarize : truncate each top k chzunks and feed all the chunks into LLM (to avoid out of input tokens)
generation : response only based on question
no_text : have no text in response 
"""

# as_chat_engine:for multi turn conversation
query_engine = index.as_query_engine(similarity_top_k=5, service_context=service_context, response_mode=response_mode, streaming=True)
response = query_engine.query("What did the author do growing up")
metadata = query_engine.get_prompts()
for k, p in metadata.items():
    print(p.get_template())
response.print_response_stream()