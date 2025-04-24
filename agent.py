from langchain.llms import Ollama
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from datetime import datetime
import json
import os

llm = Ollama(model="mistral")
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

questions = [
    "Introduce yourself.",
    "What is your experience with Python?",
    "Describe a challenging bug you’ve fixed.",
    "What’s your approach to debugging?",
]

answers_with_feedback = {}

ask_template = PromptTemplate(
    input_variables=["chat_history", "question"],
    template="""
You are a technical interviewer.

Now, ask this next question in a conversational tone:
{question}
 
Do not repeat yourself or reintroduce who you are.
Keep track of the previous questions and answers to stay coherent.

Be straight forward and don't be verbose.
""",
)

feedback_template = PromptTemplate(
    input_variables=["chat_history", "question", "answer"],
    template="""
You are an AI evaluator. Here's the interview history:

{chat_history}

The candidate answered this question:
Q: {question}
A: {answer}

Give a short, constructive comment based on their answer and prior responses in not more than 30 words.

Do not ask counter questions.
"""
)

for idx, question in enumerate(questions, 1):
    ask_chain = LLMChain(llm=llm, prompt=ask_template, memory=memory)
    bot_question = ask_chain.run(question=question)

    print(f"\nQuestion {idx}: {bot_question}")
    user_answer = input("Your Answer: ")

    memory.chat_memory.add_user_message(user_answer)

    chat_history = memory.load_memory_variables({})["chat_history"]
    feedback_chain = LLMChain(llm=llm, prompt=feedback_template)
    feedback = feedback_chain.run({"question":question, "answer":user_answer, "chat_history":chat_history})

    print(f"Feedback: {feedback}")

    answers_with_feedback[question] = {
        "answer": user_answer,
        "feedback": feedback
    }

os.makedirs("data", exist_ok=True)
with open(f"data/answers-{datetime.now()}.json", "w") as f:
    json.dump(answers_with_feedback, f, indent=4)

print("\n Assessment complete. Answers and feedback saved to data/answers.json.")