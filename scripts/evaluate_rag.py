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

wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from core.langchain_models import CachedModelFactory
from core.prompt_bank import GEN_SYSTEM, GEN_USER
from eval.metrics import AnswerCorrectness

eval_qa_dataset_df = pd.read_json("data/eval_qa_dataset.json")
swisscom_dataset_df = pd.read_json("data/swisscom_dataset.json")
swisscom_dataset_df["id"] = swisscom_dataset_df["metadata"].apply(lambda r:r["id"])

eval_data = pd.merge(eval_qa_dataset_df, swisscom_dataset_df, left_on=["relevant_id"], right_on=["id"])
eval_data = eval_data.rename(columns={"text": "context", "answer" : "ground_truth"})
eval_data = eval_data.drop(columns=["relevant_id", "metadata"])

pkg_contexts_phi = pd.read_json("data/pkg_contexts_phi_2.json").rename(columns={0:"context"})
pkg_contexts_mistral = pd.read_json("data/pkg_contexts_mistral.json").rename(columns={0:"context"})

model_factory = CachedModelFactory(llm_cache_file="cache/llm.db",
                                    embeddings_cache_dir="cache/embeddings")

critique_model = model_factory.get_chat_model("gpt-3.5-turbo-0125", **CachedModelFactory.get_default_llm_params(), max_tokens=512)
embeddings_model = model_factory.get_embedding_model("text-embedding-3-small")

print("Semantic similarity of generated contexts to the ground-truth context")
eval_embeddings = embeddings_model.embed_documents(list(eval_data["context"]))
gen_embeddings_mistral = embeddings_model.embed_documents(list(pkg_contexts_mistral["context"]))
gen_embeddings_phi = embeddings_model.embed_documents(list(pkg_contexts_phi["context"]))
print(np.mean([np.dot(e, g) for e, g in zip(eval_embeddings, gen_embeddings_phi)]))
print(np.mean([np.dot(e, g) for e, g in zip(eval_embeddings, gen_embeddings_mistral)]))


gen_model = model_factory.get_chat_model("mistralai/Mistral-7B-Instruct-v0.2", **CachedModelFactory.get_default_llm_params(), max_tokens=512)
gen_template = ChatPromptTemplate.from_messages([
    SystemMessage(GEN_SYSTEM),
    HumanMessagePromptTemplate.from_template(GEN_USER)
])
gen_chain = gen_template | gen_model | StrOutputParser()

answers = []
for context, (_, eval_row) in tqdm(zip(list(pkg_contexts_mistral["context"]), eval_data.iterrows()), total=len(eval_data)):
    answer = gen_chain.invoke({"context" : context, "question" : eval_row["question"]})
    answers.append(answer)

answer_correctness = AnswerCorrectness(llm=critique_model, embeddings=embeddings_model)
data = eval_data.copy()
data["answer"] = answers
print(data)
ac = answer_correctness.compute_score(data)
print(ac)

#TP: statements that are present in both the answer and the ground truth,
#FP: statements present in the answer but not found in the ground truth,
#FN: relevant statements found in the ground truth but omitted in the answer