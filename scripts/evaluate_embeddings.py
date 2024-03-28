import argparse
import json
import sys
from pathlib import Path
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain.vectorstores.utils import DistanceStrategy

# Support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from core.langchain_models import CachedModelFactory

def get_langchain_documents(swisscom_dataset):
    return [
        Document(
            page_content = d["text"],
            metadata = d["metadata"],
        ) for d in swisscom_dataset
    ]

def evaluate_retriever(retriever, eval_qa_dataset):
    hits = 0
    for q in eval_qa_dataset:
        docs = retriever.get_relevant_documents(q["question"])
        retrieved_ids = set([d.metadata["id"] for d in docs])
        if q["relevant_id"] in retrieved_ids:
            hits += 1
    return hits/len(eval_qa_dataset)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--swisscom_dataset", type=Path, 
                        help="Swisscom dataset JSON file.",
                        default="data/swisscom_dataset.json")
    parser.add_argument("--eval_qa_dataset", type=Path, 
                        help="Evaluation QA dataset JSON file.",
                        default="data/eval_qa_dataset.json")
    params = parser.parse_args()

    with open(params.swisscom_dataset, "r") as f:
        swisscom_dataset = json.load(f)

    with open(params.eval_qa_dataset, "r") as f:
        eval_qa_dataset = json.load(f)

    documents = get_langchain_documents(swisscom_dataset)

    model_factory = CachedModelFactory(llm_cache_file="cache/llm.db",
                                   embeddings_cache_dir="cache/embeddings")

    embedding_models = [
        "text-embedding-ada-002",
        "text-embedding-3-small",
        "text-embedding-3-large",
        "sentence-transformers/msmarco-bert-base-dot-v5",
        "BAAI/bge-large-en-v1.5",
        "WhereIsAI/UAE-Large-V1",
        "mixedbread-ai/mxbai-embed-large-v1",
        "embed-english-light-v3.0",
        "embed-english-v3.0"
    ]
    embeddings = {m : model_factory.get_embedding_model(model=m) for m in embedding_models}
    vector_stores = {m : FAISS.from_documents(documents, embeddings[m], 
                                            distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT)
                                            for m in embedding_models}

    # Cache queries in advance (batch) for faster processing
    for model in embedding_models:
        embeddings[model].pre_cache([qa["question"] for qa in eval_qa_dataset])

    top_k = 3
    retrievers = {m : vector_stores[m].as_retriever(search_kwargs={"k": top_k}) for m in embedding_models}
    retrieval_accuracy = {m : evaluate_retriever(retrievers[m], eval_qa_dataset) for m in embedding_models}
    print(list(retrieval_accuracy.keys()))
    print("\n".join([f"{v:.3f}" for v in list(retrieval_accuracy.values())]))