from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.indices.query.query_transform.base import  HyDEQueryTransform, StepDecomposeQueryTransform
from llama_index.core.query_engine import TransformQueryEngine, MultiStepQueryEngine
from llama_index.llms.openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

documents = SimpleDirectoryReader(input_files=["./data/example1/paul_graham_essay.txt"]).load_data()
# if just build index by VectorStroeIndex(documents), getting error message that exceed max input token in embedding phase
index = VectorStoreIndex.from_documents(documents)

query_engine = index.as_query_engine()

"""
HyDE : Create hypothetical answer based on initial query
"""
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

"""
summary
1. if initial query is brief, it may be more appropriate to 
use initial query instead of new query(by HyDE) to do retrieval

2. for the query that is an open-ended question, 
the hypothetical answer may have negative effect in retrieval
"""


"""
Multi-Step query transformation : decompose the initial query into subqueries
"""

index_summary = "Ready to search"
step_decompose_transform = StepDecomposeQueryTransform(llm=OpenAI(model='gpt-4'), verbose=True)
query_engine = MultiStepQueryEngine(
    query_engine=query_engine, 
    query_transform=step_decompose_transform,
    index_summary=index_summary)
# prompt = MultiStepQueryEngine.get_prompts(query_engine)
# prompt : "The original question is as follows: {query_str}\nWe have an opportunity to answer some, 
# or all of the question from a knowledge source. Context information for the knowledge source is provided 
# below, as well as previous reasoning steps.\nGiven the context and previous reasoning, return a question that
# can be answered from the context. This question can be the same as the original question, or this question can represent
# a subcomponent of the overall question.It should not be irrelevant to the original question.\nIf we cannot extract 
# more information from the context, provide 'None' as the answer. Some examples are given below: \n\nQuestion: 
# How many Grand Slam titles does the winner of the 2020 Australian Open have?\nKnowledge source context: Provides 
# names of the winners of the 2020 Australian Open\nPrevious reasoning: None\nNext question: Who was the winner of 
# the 2020 Australian Open? \n\nQuestion: Who was the winner of the 2020 Australian Open?\nKnowledge source context: 
# Provides names of the winners of the 2020 Australian Open\nPrevious reasoning: None.\nNew question: Who was the winner
# of the 2020 Australian Open? \n\nQuestion: How many Grand Slam titles does the winner of the 2020 Australian Open have?
# \nKnowledge source context: Provides information about the winners of the 2020 Australian Open\nPrevious reasoning:
# \n- Who was the winner of the 2020 Australian Open? \n- The winner of the 2020 Australian Open was Novak Djokovic.\n
# New question: None\n\nQuestion: How many Grand Slam titles does the winner of the 2020 Australian Open have?\nKnowledge
# source context: Provides information about the winners of the 2020 Australian Open - includes biographical information 
# for each winner\nPrevious reasoning:\n- Who was the winner of the 2020 Australian Open? \n- The winner of the 2020 Australian
#  Open was Novak Djokovic.\nNew question: How many Grand Slam titles does Novak Djokovic have? \n\nQuestion: {query_str}\nKnowledge
# source context: {context_str}\nPrevious reasoning: {prev_reasoning}\nNew question: 

response = query_engine.query("Who was in the first batch of the accelerator program the author started")
print(response)