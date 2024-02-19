"""summary index (list index)
index : simply stores nodes as a sequential chain.
query : during query time, loads all Nodes (if no settings like top-k, keyword filtering)
in the list into response synthesis module
"""

"""vector store index
index : store each Node and a corrsponding embedding in a vector store
query : querying a vector store index involves the top-k most similar Nodes
 and pass into response synthesis module
"""


"""tree index
index : builds a hierarchical tree from a set of Nodes 
query : querying a tree index involves traversing from root Nodes down to leaf Nodes.
by default "child_branch_factor=1" means will choose one child Node from root, if we set 
"child_branch_factor=2", then we will choose two child Nodes from root
"""


"""keyword table index
index : extracts keywords from each Node and builds a mapping from each keyword to the 
corresponding Nodes of that keyword
query : in query time, we extract relevant keywords from the query, and match 
those with pre-extracted Node keywords to get the corresponding Nodes, The extracted Nodes
then passed to response synthesis module
"""


"""knowledge graph index
background :
each node corresponds to a  unique entity and is identified by a unique identifier
each edge represents the relatonship between two nodes
triplet is a basic unit of data in the graph, for example (Subject) — [Predicate] -> (Object)
index :
query : 
"""
from llama_index import StorageContext
from llama_index.core import SimpleDirectoryReader, KnowledgeGraphIndex, SimpleGraphStore
from llama_index.llms.openai import OpenAI
documents = SimpleDirectoryReader("./data").lead_data()
llm = OpenAI(temperature=0, model="text-davinci-002")
graph_store = SimpleGraphStore()
storage_context = StorageContext.from_defaults(graph_store=graph_store)
index = KnowledgeGraphIndex.from_documents(documents, max_triplets_per_chunk=2, storage_context=storage_context)
