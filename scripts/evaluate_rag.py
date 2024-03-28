import sys
from pathlib import Path
import json
import pandas as pd
from tqdm import tqdm
import numpy as np

from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate
)
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain.vectorstores.utils import DistanceStrategy
from langchain.schema import Document

wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from core.langchain_models import CachedModelFactory
from core.prompt_bank import GEN_SYSTEM, GEN_USER, NO_RAG_SYSTEM, NO_RAG_USER
from eval.metrics import AnswerCorrectness

def get_langchain_documents(swisscom_dataset):
    return [
        Document(
            page_content = d["text"],
            metadata = d["metadata"],
        ) for d in swisscom_dataset
    ]

with open("data/swisscom_dataset.json", "r") as f:
    swisscom_dataset = json.load(f)
documents = get_langchain_documents(swisscom_dataset)

swisscom_dataset_df = pd.read_json("data/swisscom_dataset.json")
eval_qa_dataset_df = pd.read_json("data/eval_qa_dataset.json")
swisscom_dataset_df["id"] = swisscom_dataset_df["metadata"].apply(lambda r:r["id"])

eval_data = pd.merge(eval_qa_dataset_df, swisscom_dataset_df, left_on=["relevant_id"], right_on=["id"])
eval_data = eval_data.rename(columns={"text": "context", "answer" : "ground_truth"})
eval_data = eval_data.drop(columns=["relevant_id", "metadata"])

pkg_contexts_phi = pd.read_json("data/pkg_contexts_phi_2.json").rename(columns={0:"context"})
pkg_contexts_mistral = pd.read_json("data/pkg_contexts_mistral.json").rename(columns={0:"context"})

model_factory = CachedModelFactory(llm_cache_file="cache/llm.db",
                                    embeddings_cache_dir="cache/embeddings")

critique_model = model_factory.get_chat_model("gpt-3.5-turbo-0125", **CachedModelFactory.get_default_llm_params(), max_tokens=1024)
critique_embedding = model_factory.get_embedding_model("text-embedding-3-large")

embeddings_model_where = model_factory.get_embedding_model("WhereIsAI/UAE-Large-V1")
embeddings_model_openai = model_factory.get_embedding_model("text-embedding-3-large")

print("Semantic similarity of generated contexts to the ground-truth context")
eval_embeddings = critique_embedding.embed_documents(list(eval_data["context"]))
gen_embeddings_mistral = critique_embedding.embed_documents(list(pkg_contexts_mistral["context"]))
gen_embeddings_phi = critique_embedding.embed_documents(list(pkg_contexts_phi["context"]))
print(np.mean([np.dot(e, g) for e, g in zip(eval_embeddings, gen_embeddings_phi)]))
print(np.mean([np.dot(e, g) for e, g in zip(eval_embeddings, gen_embeddings_mistral)]))


embedding_vector_store_where = FAISS.from_documents(documents, embeddings_model_where, 
                                            distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT)
embeddings_retriever = embedding_vector_store_where.as_retriever(search_kwargs={"k": 3})

embedding_vector_store_openai = FAISS.from_documents(documents, embeddings_model_openai)
embeddings_retriever_10 = embedding_vector_store_openai.as_retriever(search_kwargs={"k": 10})

bm25_retriever = BM25Retriever.from_documents(documents, k=3)
fusion_retriever = EnsembleRetriever(
    retrievers=[embeddings_retriever, bm25_retriever], weights=[0.6, 0.4]
)

gen_model = model_factory.get_chat_model("gpt-3.5-turbo-0125", **CachedModelFactory.get_default_llm_params(), max_tokens=512)
gen_model_gpt4 = model_factory.get_chat_model("gpt-4-0125-preview", **CachedModelFactory.get_default_llm_params(), max_tokens=512)
gen_model_mistral_small = model_factory.get_chat_model("mistral-small-latest", **CachedModelFactory.get_default_llm_params(), max_tokens=512)
gen_model_mistral = model_factory.get_chat_model("mistralai/Mistral-7B-Instruct-v0.2", **CachedModelFactory.get_default_llm_params(), max_tokens=512)
gen_model_mixtral = model_factory.get_chat_model("mistralai/Mixtral-8x7B-Instruct-v0.1", **CachedModelFactory.get_default_llm_params(), max_tokens=512)
gen_template = ChatPromptTemplate.from_messages([
    SystemMessage(GEN_SYSTEM),
    HumanMessagePromptTemplate.from_template(GEN_USER)
])
no_rag_template = ChatPromptTemplate.from_messages([
    SystemMessage(NO_RAG_SYSTEM),
    HumanMessagePromptTemplate.from_template(NO_RAG_USER)
])
gen_chain = gen_template | gen_model | StrOutputParser()
gen_chain_gpt4 = gen_template | gen_model_gpt4 | StrOutputParser()
gen_chain_mistral_small = gen_template | gen_model_mistral_small | StrOutputParser()
gen_chain_mistral = gen_template | gen_model_mistral | StrOutputParser()
gen_chain_mixtral = gen_template | gen_model_mixtral | StrOutputParser()
gen_chain_no_rag = no_rag_template | gen_model | StrOutputParser()

answers = []
for context, (_, eval_row) in tqdm(zip(list(pkg_contexts_mistral["context"]), eval_data.iterrows()), total=len(eval_data)):
    try:
        answer = gen_chain.invoke({"context" : context, "question" : eval_row["question"]})
    except Exception as e:
        answer = ""
        print(context)
    answers.append(answer)

answer_correctness = AnswerCorrectness(llm=critique_model, embeddings=critique_embedding, alpha=0.7)
data = eval_data.copy()
data["answer"] = answers
print("PKG Mistral")
print(f"{answer_correctness.compute_score(data):.3f}")

'''
print("Examples")
q = "I'm planning a business trip to the UK and will need to use a lot of data. What happens if I exceed 40 GB of data usage while abroad with my Enterprise Mobile Global subscription?"
docs = fusion_retriever.get_relevant_documents(q)
context = "\n".join([d.page_content for d in docs])

print("Fusion + GPT3.5-Turbo")
print(gen_chain.invoke({"context" : context, "question" : q}))

print("Fusion + mistral-small")
print(gen_chain_mistral_small.invoke({"context" : context, "question" : q}))

print("No RAG + GPT-3.5")
print(gen_chain_no_rag.invoke({"question" : q}))
exit()
'''

contexts = []
answers = []
for _, eval_row in tqdm(eval_data.iterrows(), total=len(eval_data)):
    q = eval_row["question"]
    docs = embeddings_retriever.get_relevant_documents(q)
    context = "\n".join([d.page_content for d in docs])
    answer = gen_chain.invoke({"context" : context, "question" : q})
    contexts.append(context)
    answers.append(answer)
data = eval_data.copy()
data["context"] = contexts
data["answer"] = answers
print("WhereIsAI/UAE-Large-V1 (K=3) + gpt3.5")
print(f"{answer_correctness.compute_score(data):.3f}")

contexts = []
answers = []
for _, eval_row in tqdm(eval_data.iterrows(), total=len(eval_data)):
    q = eval_row["question"]
    docs = embeddings_retriever_10.get_relevant_documents(q)
    context = "\n".join([d.page_content for d in docs])
    answer = gen_chain.invoke({"context" : context, "question" : q})
    contexts.append(context)
    answers.append(answer)
data = eval_data.copy()
data["context"] = contexts
data["answer"] = answers
print("openai large (K=10) + gpt3.5")
print(f"{answer_correctness.compute_score(data):.3f}")

contexts = []
answers = []
for _, eval_row in tqdm(eval_data.iterrows(), total=len(eval_data)):
    q = eval_row["question"]
    docs = embedding_vector_store_where.similarity_search_with_score(q, k=50)
    docs = list(filter(lambda x:x[1]>=0.6, docs))
    if len(docs) > 3:
        docs = docs[:3]
    context = "\n".join([d[0].page_content for d in docs])
    answer = gen_chain.invoke({"context" : context, "question" : q})
    contexts.append(context)
    answers.append(answer)
data = eval_data.copy()
data["context"] = contexts
data["answer"] = answers
print("threshold WhereIsAI/UAE-Large-V1 (K=3) + gpt3.5")
print(f"{answer_correctness.compute_score(data):.3f}")

contexts = []
answers = []
for _, eval_row in tqdm(eval_data.iterrows(), total=len(eval_data)):
    q = eval_row["question"]
    context = eval_row["context"]
    answer = gen_chain.invoke({"context" : context, "question" : q})
    contexts.append(context)
    answers.append(answer)
data = eval_data.copy()
data["context"] = contexts
data["answer"] = answers
print("relevant context + gpt3.5")
print(f"{answer_correctness.compute_score(data):.3f}")

contexts = []
answers = []
for _, eval_row in tqdm(eval_data.iterrows(), total=len(eval_data)):
    q = eval_row["question"]
    context = eval_row["context"]
    answer = gen_chain_gpt4.invoke({"context" : context, "question" : q})
    contexts.append(context)
    answers.append(answer)
data = eval_data.copy()
data["context"] = contexts
data["answer"] = answers
print("relevant context + gpt4")
print(f"{answer_correctness.compute_score(data):.3f}")

contexts = []
answers = []
for _, eval_row in tqdm(eval_data.iterrows(), total=len(eval_data)):
    q = eval_row["question"]
    docs = fusion_retriever.get_relevant_documents(q)
    context = "\n".join([d.page_content for d in docs])
    answer = gen_chain.invoke({"context" : context, "question" : q})
    contexts.append(context)
    answers.append(answer)
data = eval_data.copy()
data["context"] = contexts
data["answer"] = answers
print("fusion retriever (BM25, WhereIsAI/UAE-Large-V1) + gpt3.5")
print(f"{answer_correctness.compute_score(data):.3f}")

contexts = []
answers = []
for _, eval_row in tqdm(eval_data.iterrows(), total=len(eval_data)):
    q = eval_row["question"]
    docs = fusion_retriever.get_relevant_documents(q)
    context = "\n".join([d.page_content for d in docs])
    answer = gen_chain_gpt4.invoke({"context" : context, "question" : q})
    contexts.append(context)
    answers.append(answer)
data = eval_data.copy()
data["context"] = contexts
data["answer"] = answers
print("fusion retriever (BM25, WhereIsAI/UAE-Large-V1) + gpt4turbo")
print(f"{answer_correctness.compute_score(data):.3f}")

contexts = []
answers = []
for _, eval_row in tqdm(eval_data.iterrows(), total=len(eval_data)):
    q = eval_row["question"]
    docs = fusion_retriever.get_relevant_documents(q)
    context = "\n".join([d.page_content for d in docs])
    answer = gen_chain_mixtral.invoke({"context" : context, "question" : q})
    contexts.append(context)
    answers.append(answer)
data = eval_data.copy()
data["context"] = contexts
data["answer"] = answers
print("fusion retriever (BM25, WhereIsAI/UAE-Large-V1) + mixtral")
print(f"{answer_correctness.compute_score(data):.3f}")

contexts = []
answers = []
for _, eval_row in tqdm(eval_data.iterrows(), total=len(eval_data)):
    q = eval_row["question"]
    docs = fusion_retriever.get_relevant_documents(q)
    context = "\n".join([d.page_content for d in docs])
    answer = gen_chain_mistral_small.invoke({"context" : context, "question" : q})
    contexts.append(context)
    answers.append(answer)
data = eval_data.copy()
data["context"] = contexts
data["answer"] = answers
print("fusion retriever (BM25, WhereIsAI/UAE-Large-V1) + mistral-small")
print(f"{answer_correctness.compute_score(data):.3f}")

contexts = []
answers = []
for _, eval_row in tqdm(eval_data.iterrows(), total=len(eval_data)):
    q = eval_row["question"]
    docs = fusion_retriever.get_relevant_documents(q)
    context = "\n".join([d.page_content for d in docs])
    answer = gen_chain_mistral.invoke({"context" : context, "question" : q})
    contexts.append(context)
    answers.append(answer)
data = eval_data.copy()
data["context"] = contexts
data["answer"] = answers
print("fusion retriever (BM25, WhereIsAI/UAE-Large-V1) + mistral-7b")
print(f"{answer_correctness.compute_score(data):.3f}")

contexts = []
answers = []
for _, eval_row in tqdm(eval_data.iterrows(), total=len(eval_data)):
    q = eval_row["question"]
    answer = gen_chain_no_rag.invoke({"question" : q})
    contexts.append("")
    answers.append(answer)
data = eval_data.copy()
data["context"] = contexts
data["answer"] = answers
print("No rag + gpt3.5turbo")
print(f"{answer_correctness.compute_score(data):.3f}")