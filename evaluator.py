import os
import json
from langchain.llms import Ollama
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

with open("xyz.json", "r") as f:
    json_data = json.load(f)

json_str = json.dumps(json_data, indent=2)

llm = Ollama(model="mistral")

evaluator_template = PromptTemplate(
    input_variables=["json"],
    template="""
        You are a recruiter.
        Given the following JSON data of assessments taken by different candidates with unique id:
        {json}
        Rank all the candidates and provide blunt reason for each rank.
        Output format will be a list with parameters: id, rank, reason.
        Be strict to the output format and don't be verbose.
    """,
)

evaluator_chain = LLMChain(llm=llm, prompt=evaluator_template)
response = evaluator_chain.run(json=json_str)

print(response)




