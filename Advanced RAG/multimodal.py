from llama_index.core.response.notebook_utils import display_source_node
from llama_index.core.schema import ImageNode
from llama_index.multi_modal_llms.openai import OpenAIMultiModal
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import SimpleDirectoryReader
from llama_index.core.indices import MultiModalVectorStoreIndex
from llama_index.core.query_engine import SimpleMultiModalQueryEngine
from llama_index.core import StorageContext
from PIL import Image
from dotenv import load_dotenv
import os
import matplotlib.pyplot as plt
import chromadb

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

def plot_images(image_paths):
    images_shown = 0
    plt.figure(figsize=(16, 9))
    for img_path in image_paths:
        if os.path.isfile(img_path):
            image = Image.open(img_path)
            plt.subplot(2, 3, images_shown + 1)
            plt.imshow(image)
            plt.xticks([])
            plt.yticks([])
            images_shown += 1
            if images_shown >= 9:
                break


chroma_client = chromadb.PersistentClient(path="./testing_chroma")
text_chroma_collection = chroma_client.get_or_create_collection("testing_text")
img_chroma_collection = chroma_client.get_or_create_collection("testing_image")
# create chroma collection to store embeddings of text
text_store = ChromaVectorStore(chroma_collection=text_chroma_collection)
# create chroma collection to store embeddings of images
image_store = ChromaVectorStore(chroma_collection=img_chroma_collection)
storage_context  = StorageContext.from_defaults(vector_store=text_store, image_store=image_store)

# read all the text and image file
documents = SimpleDirectoryReader("./multi-modal-data/").load_data()
index = MultiModalVectorStoreIndex.from_documents(
    documents,
    storage_context=storage_context,
)

retriever = index.as_retriever(
    similarity_top_k=3, image_similarity_top_k=3
)

retrieval_text_images = retriever.retrieve("Tell me about Tesla")

# showing retrieved text and images
retrieved_image = []
for node in retrieval_text_images:
    if isinstance(node.node, ImageNode):
        retrieved_image.append(node.node.metadata["file_path"])
    else:
        display_source_node(node, source_length=200)
plot_images(retrieved_image)

# plug retriever into query engine (passing retrieved nodes into GPT4V for final answer)
query_engine = SimpleMultiModalQueryEngine(retriever=retriever)
query_str = "Tell me about Tesla"
response = query_engine.query(query_str)
print(response)