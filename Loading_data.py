"""
loaders (data connectors) ingest data 
from different data sources and format data into Document
"""
# Loading with SimpleDirectoryReader
# can read Markdown, PDF, Word, PPT, image,....
from llama_index import SimpleDirectoryReader
documents = SimpleDirectoryReader("./data").load_data()

# Using readers from LlamaHub
# PPT
from pathlib import Path
from llama_index import download_loader
PptxReader = download_loader("PptxReader")
loader = PptxReader()
documents = loader.load_data(file=Path('./deck.pptx'))

# image
from pathlib import Path
from llama_index import download_loader
ImageReader = download_loader("ImageReader")
# If the Image has key-value pairs text, use text_type = "key_value"
loader = ImageReader(text_type = "key_value")
documents = loader.load_data(file=Path('./receipt.png'))
# If the Image has plain text, use text_type = "plain_text"
loader = ImageReader(text_type = "plain_text", model_kwargs=dict(lang="deu+eng"))
documents = loader.load_data(file=Path('./image.png'))

# Creating document 
from llama_index.core.schema import Document, TextNode
doc = Document(text='text')

"""
After loading data, we need to transform our data like chunking,
extracting metadata, and embedding chunks

To do the transformations, our input will be Node objects 
(Document us subclass of Node), and output will be Node, too
"""
from llama_index import VectorStoreIndex, ServiceContext
from llama_index.core.text_splitter  import SentenceSplitter

# customize chunk splitter
# splitter such as HTMLsplitter, SentenceWindowNodeParser, JSONNodeParser
text_splitter = SentenceSplitter(chunk_size=512, chunk_overlap=30)
service_context = ServiceContext.from_defaults(text_splitter=text_splitter)
index = VectorStoreIndex.from_documents(documents, service_context=service_context)
index.as_query_engine()

"""
we can set metadata (like file_name) that are invisible 
to the LLM and embedding model
"""
# for LLM
documents.excluded_llm_metadata_keys = ["file_name"]
# check if file_name is invisible to LLM
from llama_index.core.schema import MetadataMode
print(documents.get_content(metadata_mode=MetadataMode.LLM))
# for embedding model
documents.excluded_embed_metadata_keys = ["file_name"]
# check if file_name is invisible to embedding model
from llama_index.core.schema  import MetadataMode
print(documents.get_content(metadata_mode=MetadataMode.EMBED))

# customized metadata format
# When concatenating all key/value fields of your metadata,
# this field controls the separator between each key/value pair.
documents.metadata_seperator = '\n'

# This attribute controls how each key/value pair in your metadata
# is formatted. The two variables key and value string keys are required
documents.metadata_template = "{key}=>{value}"

# this templates controls what that metadata looks like when joined with 
# the text content of your document/node.
documents.text_template = "Metadata: {metadata_str}\n-----\nContent: {content}"

"""
we can use LLM to extract metadata from each Node
"""
from llama_index.llms import OpenAI
from llama_index.core.extractors import (
    SummaryExtractor,
    QuestionsAnsweredExtractor,
)
node_parser = SentenceSplitter(chunk_size=512)
base_nodes = node_parser.get_nodes_from_documents(documents)
for idx, node in enumerate(base_nodes):
    node.id_ = f"node-{idx}"
llm = OpenAI(temperature=0.1, model="gpt-3.5-turbo", max_tokens=512)
extractors = [
    SummaryExtractor(llm=llm, summaries=['self'], show_progress=True),
    QuestionsAnsweredExtractor(llm=llm, questions=5, show_progress=True)
]
node_to_metadata = {}
for extractor in extractors:
    metadata_dicts = extractor.extract(base_nodes)
    print(metadata_dicts)
    for node, metadata in zip(base_nodes, metadata_dicts):
        #add Metadata
        if node.node_id not in node_to_metadata:
            node_to_metadata[node.node_id] = metadata
        else:
            node_to_metadata[node.node_id].update(metadata)