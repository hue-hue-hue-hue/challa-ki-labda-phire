import os
from pathway.xpacks.llm.vector_store import VectorStoreClient
from langchain_groq import ChatGroq
from ragas.llms import LangchainLLMWrapper
import getpass
import json
from ragas import evaluate
from langchain_ollama import ChatOllama

if "GROQ_API_KEY" not in os.environ:
    os.environ["GROQ_API_KEY"] = getpass.getpass("Enter your Groq API key: ")
llm = ChatGroq(
    model="mixtral-8x7b-32768",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)
# llm = ChatOllama(model="llama3.1")
evaluator_llm = LangchainLLMWrapper(llm)


PATHWAY_PORT = 8765
PATHWAY_HOST = "127.0.0.1"

client = VectorStoreClient(
    host=PATHWAY_HOST,
    port=PATHWAY_PORT,
)

with open("query_context_answer.json") as f:
    data = json.load(f)


metrics_dict: dict = {
    "user_input": [],
    "response": [],
    "retrieved_contexts": [],
    "reference": [],    
}
for i, item in enumerate(data):
    context = ""
    question = item["question"]
    docs = client(question)
    retireved_contexts = []
    for doc in docs:
        context += doc["text"] + "/n/n"
        retireved_contexts.append(doc["text"])
    prompt = """You are a helpful chat assistant that helps answer query based on the given context.
    You will answer queries only on the basis of the following information: {context}
    Do not use outside knowledge to answer the query. If the answer is not contained in the provided information, just say that you don't know, don't try to make up an answer.
    """
    messages = [
        (
            "system",
            prompt.format(context=context),
        ),
        ("human", question),
    ]
    ai_msg = llm.invoke(messages)
    answer = ai_msg.content
    
    metrics_dict["user_input"].append(question)
    metrics_dict["response"].append(answer)
    metrics_dict["retrieved_contexts"].append(retireved_contexts)
    # check if supporting facts is in the data
    if "supporting_facts" in item:
        metrics_dict["reference"].append(" ".join(item["supporting_facts"]))
    else:
        metrics_dict["reference"].append([])



from ragas.metrics import (
    LLMContextRecall,
    LLMContextPrecisionWithReference,
    ContextPrecision,
    ContextEntityRecall,
    Faithfulness,
    ResponseRelevancy
)
from ragas import evaluate
from datasets import Dataset

metrics = [
    LLMContextRecall(),
    LLMContextPrecisionWithReference(),
    ContextPrecision(),
    ContextEntityRecall(),
    Faithfulness(),
    ResponseRelevancy(),
]

dataset = Dataset.from_dict(metrics_dict)
metrics_dict = evaluate(
    llm=evaluator_llm,
    dataset=dataset,
    metrics=metrics,
)

print(metrics_dict)

