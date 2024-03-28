import sys
from pathlib import Path
import json
import pandas as pd
from tqdm import tqdm
import numpy as np
import random

from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate
)
from langchain_core.messages import AIMessage
from langchain_core.messages import SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain.schema import Document
from datasets import load_dataset

wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from core.langchain_models import CachedModelFactory
from core.prompt_bank import GEN_SYSTEM, GEN_USER
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

model_factory = CachedModelFactory(llm_cache_file="cache/llm.db",
                                    embeddings_cache_dir="cache/embeddings")

gen_model_names = ["mistralai/Mixtral-8x7B-Instruct-v0.1", "gpt-3.5-turbo-0125", "mistral-small-latest", "mistralai/Mistral-7B-Instruct-v0.2", "claude-3-haiku-20240307", "gpt-4-0125-preview"]
gen_models = {m : model_factory.get_chat_model(m, **CachedModelFactory.get_default_llm_params(), max_tokens=512) for m in gen_model_names}
critique_model = model_factory.get_chat_model("gpt-3.5-turbo-0125", **CachedModelFactory.get_default_llm_params(), max_tokens=512)

gen_template = ChatPromptTemplate.from_messages([
    SystemMessage(GEN_SYSTEM),
    HumanMessagePromptTemplate.from_template(GEN_USER)
])
gen_chains = {m : gen_template | gen_models[m] | StrOutputParser() for m in gen_model_names}

no_match_dataset = load_dataset("web_questions")
random.seed(53)
no_match_dataset = random.sample(list(no_match_dataset["train"]["question"]), 350)

IRRELEVANT_SYSTEM = "Classify if the model rejects answering the question. `yes` for rejection, `no` if the model answers."
IRRELEVANT_USER = """Model answer: {answer}
Class: """
IRRELEVANT_EXAMPLE_YES = "I cannot provide help with that question."
IRRELEVANT_EXAMPLE_NO = "I can only help with Swisscom related queries. The speed of light is 299 792 458 m / s."
cannot_help_prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content=IRRELEVANT_SYSTEM),
    HumanMessagePromptTemplate.from_template(IRRELEVANT_USER).format(answer=IRRELEVANT_EXAMPLE_YES),
    AIMessage(content="yes"),
    HumanMessagePromptTemplate.from_template(IRRELEVANT_USER).format(answer=IRRELEVANT_EXAMPLE_NO),
    AIMessage(content="no"),
    HumanMessagePromptTemplate.from_template(IRRELEVANT_USER)
])
classify_chain = cannot_help_prompt | critique_model | StrOutputParser()

for model_name in gen_model_names:
    cnt = 0
    for question in tqdm(no_match_dataset):
        answer = gen_chains[model_name].invoke({"question" : question, "context" : ""})
        class_reject = classify_chain.invoke({"answer" : answer})
        if "yes" in class_reject.lower():
            cnt += 1
    print(f"{model_name}: {(cnt / len(no_match_dataset)):.3f}")

