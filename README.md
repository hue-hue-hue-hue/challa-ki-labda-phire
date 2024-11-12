# Experiments

## Setup
Each script has its own usage and thus requires different .env
However most only need `OPENAI_API_KEY`

WARNING: Kindly monitor OpenAI credits while using RAGAS

## Directory Structure
```
.
├── .env.sample                     
├── .gitignore                   
├── client.py                    
├── convert_pdf_to_md.py                     was used to convert the pdfs to markdown files using DoclingV2         
├── Dockerfile
├── docling_parser.py                        pathway implementation of docling(incomplete due to futures/concurrent error)
├── markdown_splitter.py                     pathway implementation of markdown splitter, uses a combination of markdown and semantic splitter
├── query_context_answer_example.json        example json, which is used to calculate the ragas scores
├── ragas_test_bench.py                      script to calculate the ragas score of RAG
├── README.md 
├── requirements.txt 
├── semantic_splitter.py                     pathway implementation of semantic splitter(improved llama_index version)
├── server.py
├── __init__.py
├── docling_tatdqa                           markdown files parsed from docling 
└── finance
    ├── financebench                         dataset
    └── ragas_scores                         csv files containing detailed ragas score for all experiments
```

## Approaches
- For testing we used RAGAS to evaluate choice in the elements: embedding model, splitting strategy and parser. 
- RAGAS evaluates relevance, accuracy, granularity, adaptability, and sufficiency of generated responses. It ensures the model retrieves relevant information and generates precise, detailed, and complete answers across various contexts.