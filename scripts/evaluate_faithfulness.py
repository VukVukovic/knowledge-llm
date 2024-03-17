import sys
from pathlib import Path
import json
import pandas as pd
from tqdm import tqdm

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
from eval.metrics import Faithfulness, AnswerCorrectness

eval_qa_dataset_df = pd.read_json("data/eval_qa_dataset.json")
swisscom_dataset_df = pd.read_json("data/swisscom_dataset.json")
swisscom_dataset_df["id"] = swisscom_dataset_df["metadata"].apply(lambda r:r["id"])

eval_data = pd.merge(eval_qa_dataset_df, swisscom_dataset_df, left_on=["relevant_id"], right_on=["id"])
eval_data = eval_data.rename(columns={"text": "context", "answer" : "ground_truth"})
eval_data = eval_data.drop(columns=["relevant_id", "metadata"])

model_factory = CachedModelFactory(llm_cache_file="cache/llm.db",
                                    embeddings_cache_dir="cache/embeddings")

default_params = CachedModelFactory.get_default_llm_params()
full_params_models = ["gemini-pro", "mistralai/Mistral-7B-Instruct-v0.2", "mistralai/Mixtral-8x7B-Instruct-v0.1", 
                   "zero-one-ai/Yi-34B-Chat", "Qwen/Qwen1.5-72B-Chat", "mistral-small-latest",
                   "claude-3-haiku-20240307", "gpt-3.5-turbo-0125", 
                   "mistral-medium-latest", "claude-3-sonnet-20240229"]
gen_models = {model_name:model_factory.get_chat_model(model_name, max_tokens=512, 
                                                      **CachedModelFactory.get_default_llm_params())
              for model_name in full_params_models}

gen_template = ChatPromptTemplate.from_messages([
    SystemMessage(GEN_SYSTEM),
    HumanMessagePromptTemplate.from_template(GEN_USER)
])

gen_chains = {m : gen_template | gen_models[m] | StrOutputParser() for m in gen_models}

critique_model = model_factory.get_chat_model("gpt-3.5-turbo-0125", max_tokens=1024, **default_params)

faithfulness = Faithfulness(llm=critique_model)
for model_name, chain in gen_chains.items():
    print(model_name)
    answers = []
    for i, e in tqdm(eval_data.iterrows(), total=len(eval_data)):
        answers.append(chain.invoke({"context" : e["context"], 
                                     "question" : e["question"]}))
    data = eval_data.copy()
    data["answer"] = answers
    print(model_name, faithfulness.compute_score(data))    

