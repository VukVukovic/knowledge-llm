import json
import sys
import random
import numpy as np
from pathlib import Path
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain.vectorstores.utils import DistanceStrategy
from datasets import load_dataset
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

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

def get_accuracy(retriever, eval_qa_dataset):
    hits = 0
    for q in eval_qa_dataset:
        docs = retriever.get_relevant_documents(q["question"])
        retrieved_ids = set([d.metadata["id"] for d in docs])
        if q["relevant_id"] in retrieved_ids:
            hits += 1
    return hits/len(eval_qa_dataset)

def get_no_match_accuracy(retriever, no_match_dataset):
    no_matches = 0
    for q in no_match_dataset:
        docs = retriever.get_relevant_documents(q)
        print(q)
        print(docs)
        break
        if len(docs) == 0:
            no_matches += 1
    return no_matches/len(no_match_dataset)


with open("data/swisscom_dataset.json", "r") as f:
    swisscom_dataset = json.load(f)

with open("data/eval_qa_dataset.json", "r") as f:
    eval_qa_dataset = json.load(f)

no_match_dataset = load_dataset("web_questions")
random.seed(53)
no_match_dataset = random.sample(list(no_match_dataset["train"]["question"]), 350)

documents = get_langchain_documents(swisscom_dataset)

model_factory = CachedModelFactory(llm_cache_file="cache/llm.db",
                                embeddings_cache_dir="cache/embeddings")


embeddings = model_factory.get_embedding_model(model="text-embedding-3-small")
embeddings.pre_cache(no_match_dataset)

vector_store = FAISS.from_documents(documents, embeddings, 
                                        distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT)

x = []
y = []
for threshold in [0.6]:
    retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"score_threshold": threshold}
    )
    accuracy = get_accuracy(retriever, eval_qa_dataset)
    no_match_accuracy = get_no_match_accuracy(retriever, no_match_dataset)
    x.append(accuracy)
    y.append(no_match_accuracy)

plt.plot(x, y)
plt.show()