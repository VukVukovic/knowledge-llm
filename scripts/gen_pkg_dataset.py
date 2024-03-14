import argparse
import sys
import json
import jsonlines
from pathlib import Path
from tqdm import tqdm
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain_core.messages import SystemMessage
from langchain_core.output_parsers import StrOutputParser

# Support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from core.prompt_bank import PKG_SYSTEM, PKG_USER
from core.langchain_models import CachedModelFactory

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--swisscom_dataset", type=Path, 
                        help="Swisscom dataset JSON file.",
                        default="data/swisscom_dataset.json")
    parser.add_argument("--pkg_dataset", type=Path, 
                        help="Path to save PKG dataset in JSON file.",
                        default="data/pkg_dataset.json")
    params = parser.parse_args()

    model_factory = CachedModelFactory(llm_cache_file="cache/llm.db",
                                    embeddings_cache_dir="cache/embeddings")
    
    model = model_factory.get_chat_model("mistralai/Mixtral-8x7B-Instruct-v0.1", 
                                         **CachedModelFactory.get_default_llm_params())

    with open(params.swisscom_dataset, "r") as f:
        dataset = json.load(f)

    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=PKG_SYSTEM),
        HumanMessagePromptTemplate.from_template(PKG_USER)
    ])
    pkg_gen_chain = prompt | model | StrOutputParser()

    pkg_dataset = []
    for sample in tqdm(dataset):
        try:
            pkg = pkg_gen_chain.invoke({
                "context" : sample["text"]
            })
            pkg_dataset.append({
                "input" : pkg.strip(),
                "output" : sample["text"],
                "id" : sample["metadata"]["id"]
            })
        except Exception as e:
            print(f'Skipping {sample["metadata"]["id"]} due to `{e}`.')
            continue
    
    with open("data/pkg_together.jsonl", "w") as f:
        for d in pkg_dataset:
            json.dump({
                "text" : f"Question: {d['input']}\nContext: {d['output']}"
            }, f)
            f.write("\n")

    with open(params.pkg_dataset, "w") as f:
        json.dump(pkg_dataset, f, indent=1)