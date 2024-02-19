from llama_index.core.node_parser import SentenceWindowNodeParser,  SentenceSplitter
from llama_index.core.postprocessor import MetadataReplacementPostProcessor
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.llms.openai import OpenAI
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from dotenv import load_dotenv
import os

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

node_parser = SentenceWindowNodeParser.from_defaults(
    # specify how many sentences of each side to keep (default  5)
    window_size=3,
    # set the name of metadata key to keep surroudings sentences
    window_metadata_key="surroudings",
    # the metadata key that holds the original sentence
    original_text_metadata_key="target",
)

text_splitter = SentenceSplitter()
llm = OpenAI(model="gpt-3.5-turbo", temperature=0.1)
embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-mpnet-base-v2", max_length=512)

# set to default
Settings.llm = llm
Settings.embed_model = embed_model
Settings.text_splitter = text_splitter

documents = SimpleDirectoryReader(
    input_files=["./data/IPCC_AR6_WGII_Chapter03.pdf"]
).load_data()

# load data
node = node_parser.get_nodes_from_documents(documents)
base_nodes = text_splitter.get_nodes_from_documents(documents)

# build index 
sentence_index = VectorStoreIndex(node)
base_index = VectorStoreIndex(base_nodes)

# query
sentence_query_engine = sentence_index.as_query_engine(
    similarity_top_k=2, 
    node_postprocessors=
    [MetadataReplacementPostProcessor(target_metadata_key="surroudings")]
)

base_query_engine = base_index.as_query_engine(
    similarity_top_k=4
)

# base_response will consume more tokens than sentence window
sentence_response = sentence_query_engine.query("What are the concerns surrounding the AMOC?")
base_response = base_query_engine.query("What are the concerns surrounding the AMOC?")

# results comparison
print("Sentence Window : \n", sentence_response)
print("-------------------------\nBase : \n", base_response)

for source_node in sentence_response.source_nodes:
    print("\n--------")
    print(source_node.node.metadata["target"])

# there are few retrieve documents that mention AMOC since the number of token in each document is too large (bad for calculate similarity)
for node in base_response.source_nodes:
    print("\n--------")
    print("AMOC mentioned?", "AMOC" in node.node.text)
    