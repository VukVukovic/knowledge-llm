import json
import sys
import random
import numpy as np
from pathlib import Path
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain.vectorstores.utils import DistanceStrategy
from datasets import load_dataset
import matplotlib.pyplot as plt

# Support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from core.langchain_models import CachedModelFactory
from core.reranking_retriever import RerankingRetriever

def get_langchain_documents(swisscom_dataset):
    return [
        Document(
            page_content = d["text"],
            metadata = d["metadata"],
        ) for d in swisscom_dataset
    ]

def get_accuracy(scored_docs, eval_qa_dataset):
    hits = 0
    for docs, q in zip(scored_docs, eval_qa_dataset):
        retrieved_ids = set([d[0].metadata["id"] for d in list(docs)[:3]])
        if q["relevant_id"] in retrieved_ids:
            hits += 1
    return hits/len(eval_qa_dataset)

def get_no_match_accuracy(scored_docs):
    no_matches = 0
    for d in scored_docs:
        if len(list(d)) == 0:
            no_matches += 1
    return no_matches/len(scored_docs)

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

embeddings = model_factory.get_embedding_model(model="WhereIsAI/UAE-Large-V1")
embeddings.pre_cache(no_match_dataset)

vector_store = FAISS.from_documents(documents, embeddings, 
                                        distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT)

TOP_K = 100

match_docs = []
for qa in eval_qa_dataset:
    q = qa["question"]
    docs = vector_store.similarity_search_with_score(q, k=TOP_K, fetch_k=TOP_K)
    match_docs.append(docs)

no_match_docs = []
for q in no_match_dataset:
    docs = vector_store.similarity_search_with_score(q, k=TOP_K, fetch_k=TOP_K)
    no_match_docs.append(docs)

accuracies = []
no_match_accuracies = []
f1s = []

for threshold in np.arange(0.1, 0.95, 0.01):
    match_docs_filtered = [filter(lambda x:x[1]>=threshold, ds) for ds in match_docs]
    no_match_docs_filtered = [filter(lambda x:x[1]>=threshold, ds) for ds in no_match_docs]
    accuracy = get_accuracy(match_docs_filtered, eval_qa_dataset)
    no_match_accuracy = get_no_match_accuracy(no_match_docs_filtered)
    accuracies.append(accuracy)
    no_match_accuracies.append(no_match_accuracy)
    f1 = 2*accuracy*no_match_accuracy/(accuracy + no_match_accuracy)
    f1s.append((threshold, f1))
    #print(accuracy, no_match_accuracy)

best = sorted(f1s, key=lambda x:x[1], reverse=True)[0]
#plt.plot(accuracies, no_match_accuracies)
#plt.show()
best_threshold = best[0]

match_docs_filtered = [filter(lambda x:x[1]>=best_threshold, ds) for ds in match_docs]
no_match_docs_filtered = [filter(lambda x:x[1]>=best_threshold, ds) for ds in no_match_docs]
print("Best threshold", best_threshold)
print("Accuracy", get_accuracy(match_docs_filtered, eval_qa_dataset))
print("No match accuracy", get_no_match_accuracy(no_match_docs_filtered))

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

data_df = pd.DataFrame({
    "threshold" : np.arange(0.1, 0.95, 0.01).tolist() + np.arange(0.1, 0.95, 0.01).tolist(),
    "accuracy" : accuracies + no_match_accuracies,
    "Type" : ["Retrieval"] * len(accuracies) + ["No-match"] * len(no_match_accuracies)
})
sns.set_style("whitegrid")
params = {"text.usetex" : True,
          "font.family" : "serif",
          "font.serif" : ["Computer Modern Serif"]}
plt.rcParams.update(params)
sns.lineplot(data=data_df, x="threshold", y="accuracy", hue="Type")
plt.xlabel("Similarity threshold")
plt.ylabel("Accuracy")
plt.savefig("figures/threshold.svg")