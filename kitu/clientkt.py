import os
import json
from pathlib import Path

from ragas.llms import LangchainLLMWrapper
import getpass
import json
from ragas import evaluate
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from openai import OpenAI
from pathway.xpacks.llm.vector_store import VectorStoreClient
from ragas import evaluate
from datasets import Dataset


from ragas.metrics import (
    LLMContextRecall,
    LLMContextPrecisionWithReference,
    ContextPrecision,
    ContextEntityRecall,
    Faithfulness,
    ResponseRelevancy,
)

# Base configuration
BASE_DIR = Path("./finance")
QA_DATASET = BASE_DIR / "tatdqa_dataset.json"  # or .json
METRIC_INPUT = BASE_DIR / "tatdqa-qa-dataset-docling-para-l-unst.json"
llm = ChatOpenAI(model="gpt-4o-mini")
# llm = ChatOllama(model="llama3.2-vision")
evaluator_llm = LangchainLLMWrapper(llm)
PATHWAY_PORT = 8765
PATHWAY_HOST = "127.0.0.1"

client = VectorStoreClient(
    host=PATHWAY_HOST,
    port=PATHWAY_PORT,
)


def load_dataset(file_path: Path) -> list:
    """Load data from either JSON or JSONL file"""
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {file_path}")

    # Determine file type and load accordingly
    if file_path.suffix == ".jsonl":
        print(f"Loading JSONL file: {file_path}")
        data = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))

        # Save as JSON for future use
        json_path = file_path.with_suffix(".json")
        print(f"Converting to JSON: {json_path}")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

    elif file_path.suffix == ".json":
        print(f"Loading JSON file: {file_path}")
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")

    return data


def load_or_create_results1(data: list, max_samples: int = 5) -> dict:
    """Load existing results or create new ones"""
    if METRIC_INPUT.exists():
        print(f"Loading existing results from {METRIC_INPUT}")
        with open(METRIC_INPUT, "r", encoding="utf-8") as f:
            return json.load(f)

    print("Generating new results...")
    metrics_dict = {
        "user_input": [],
        "response": [],
        "retrieved_contexts": [],
        "reference": [],
    }

    for i, item in enumerate(data):
        if i == max_samples:
            break
        for j, ques in enumerate(item["questions"]):
            question = ques.get("question")
            if not isinstance(question, str):
                continue

            print(f"Processing question {i+1}/{max_samples}")

            # Get contexts from vector store
            docs = client.query(question)
            retrieved_contexts = [doc["text"] for doc in docs]
            context = "\n\n".join(retrieved_contexts)

            # Generate answer
            prompt = """You are a helpful chat assistant that helps answer query based on the given context.
            You will answer queries only on the basis of the following information: {context}
            Do not use outside knowledge to answer the query. If the answer is not contained in the provided information, just say that you don't know, don't try to make up an answer.
            """
            messages = [("system", prompt.format(context=context)), ("human", question)]
            answer = llm.invoke(messages).content

            # Collect reference/ground truth
            # if "evidence" in item:
            #     supporting_facts = [e["evidence_text"] for e in item["evidence"]]
            #     reference = " ".join(supporting_facts)
            # else:
            #     reference = ""

            # Store results
            metrics_dict["user_input"].append(question)
            metrics_dict["response"].append(answer)
            metrics_dict["retrieved_contexts"].append(retrieved_contexts)
            ans = ques.get("answer")
            if isinstance(ans, list):
                metrics_dict["reference"].append(" ".join(ans))
            else:
                metrics_dict["reference"].append(str(ans))

        # Save results
        with open(METRIC_INPUT, "w", encoding="utf-8") as f:
            json.dump(metrics_dict, f, indent=4, ensure_ascii=False)
        print(f"Results saved to {METRIC_INPUT}")

    return metrics_dict


def load_or_create_results(data: list, max_samples: int = 5) -> dict:
    """Load existing results or create new ones"""
    if METRIC_INPUT.exists():
        print(f"Loading existing results from {METRIC_INPUT}")
        with open(METRIC_INPUT, "r", encoding="utf-8") as f:
            return json.load(f)

    print("Generating new results...")
    metrics_dict = {
        "user_input": [],  # user questions
        "response": [],  # model responses
        "retrieved_contexts": [],  # retrieved contexts
        "ground_truth": [],  # reference answers
    }

    # Process your JSON data
    i = 0
    for item in data:
        for question in item["questions"]:
            context = ""
    
            # Get RAG response for this question
            retireved_contexts = []
            docs = client(question["question"])
            for doc in docs:
                context += doc["text"] + "/n/n"
                retireved_contexts.append(doc["text"])

            # Get model response using your existing prompt template
            # context_text = "\n\n".join(contexts)
            prompt = """You are a helpful chat assistant that helps answer query based on the given context.
            You will answer queries only on the basis of the following information: {context}
            Do not use outside knowledge to answer the query. If the answer is not contained in the provided information, just say that you don't know, don't try to make up an answer.
            """
            messages = [
                ("system", prompt.format(context=context)),
                ("human", question["question"]),
            ]
            response = llm.invoke(messages).content

            # Convert answer to string if it's not already
            ground_truth = (
                ", ".join(map(str, question["answer"]))
                if isinstance(question["answer"], list)
                else str(question["answer"])
            )

            # Add to metrics dictionary
            if response.lower().count("i don't know") == 0:
                metrics_dict["user_input"].append(question["question"])
                metrics_dict["response"].append(response)
                metrics_dict["retrieved_contexts"].append(retireved_contexts)
                metrics_dict["ground_truth"].append(ground_truth)
        with open(METRIC_INPUT, "w", encoding="utf-8") as f:
            json.dump(metrics_dict, f, indent=4, ensure_ascii=False)
        print(f"Results saved to {METRIC_INPUT}")

    return metrics_dict


def evaluate_rag():
    """Main evaluation function"""
    # Ensure directories exist
    BASE_DIR.mkdir(exist_ok=True)

    # Load dataset (handles both JSON and JSONL)
    test_data = load_dataset(QA_DATASET)

    # Get or create results
    metrics_dict = load_or_create_results(test_data, -1)
    mod_dict = {
        "user_input": metrics_dict["user_input"][0:50],
        "response": metrics_dict["response"][0:50],
        "retrieved_contexts": metrics_dict["retrieved_contexts"][0:50],
        "ground_truth": metrics_dict["ground_truth"][0:50],
    }

    # Convert to Dataset for Ragas
    dataset = Dataset.from_dict(mod_dict)

    # Define metrics

    from ragas.metrics import (
        AnswerCorrectness,
        AnswerSimilarity,
        ContextPrecision,
        ContextRecall,
        Faithfulness,
    )

    metrics = [
        LLMContextRecall(),
        LLMContextPrecisionWithReference(),
        Faithfulness(),
        ResponseRelevancy(),
    ]
    # metrics = [
    #     AnswerCorrectness(),  # Measures numerical accuracy
    #     AnswerSimilarity(),  # Compares answer similarity with ground truth
    #     ContextPrecision(),  # Measures relevance of retrieved context
    #     ContextRecall(),  # Measures if important info was retrieved
    #     Faithfulness(),  # Checks if answer is supported by context
    # ]
    results = []
    results = evaluate(
        llm=evaluator_llm,
        dataset=dataset,
        metrics=metrics,
    )
    df = results.to_pandas()
    df.to_json(BASE_DIR / "scoretqda-semanticmarkdown-unstruct.json")
    df.to_csv(BASE_DIR / "scoretqda-semanticmarkdown-unstruct.csv", index=False)

    # Save evaluation results
    # eval_results_file = BASE_DIR / "evaluation_results.json"
    # with open(eval_results_file, "w", encoding="utf-8") as f:
    #     json.dump(results, f, indent=4)

    return results


if __name__ == "__main__":
    results = evaluate_rag()
    print("\nEvaluation Results:")
    # print(json.dumps(results, indent=2))
