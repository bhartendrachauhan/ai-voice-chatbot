from langchain.embeddings import OllamaEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
from langchain.llms import Ollama
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import numpy as np

embedding_model = OllamaEmbeddings(model="nomic-embed-text")

text1 = "I'm a passionate software engineer who brings strong software development skills to the table. Over the years, I've worked on multiple stacks of Javascript and python which helped me in building the confidence to provide user centric tech solutions."
text2 = "I have worked on html css. I'm good with frontend."

vec1 = embedding_model.embed_query(text1)
vec2 = embedding_model.embed_query(text2)

similarity = cosine_similarity([vec1], [vec2])[0][0]

llm = Ollama(model="mistral")

evaluator_template = PromptTemplate(
    input_variables=["ideal","candidate"],
    template="""
        You are an expert interviewer.
        Compare the following two answers and evaluate the candidate's response.
        Give a score from 1 to 10 and a brief feedback.

        Ideal Answer:
        {ideal}

        Candidate's Answer:
        {candidate}

        Score and Feedback:
    """,
)

evaluator_chain = LLMChain(llm=llm, prompt=evaluator_template)
response = evaluator_chain.run(ideal=text1,candidate=text2)

print(response)
print(f"Cosine Similarity: {similarity:.4f}")