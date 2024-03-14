import argparse
import sys
import json
import random
from pathlib import Path
from tqdm import tqdm
from langchain.output_parsers.json import SimpleJsonOutputParser
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate
)
from langchain_core.messages import AIMessage

# Support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from core.prompt_bank import QA_GEN_SYSTEM_PROMPT, QA_GEN_SYSTEM_COMMUNITY_PROMPT, QA_GEN_USER_PROMPT, QA_GEN_USER_PROMPT, QA_GEN_SYSTEM_FORMAT_PROMPT, QA_GEN_EXAMPLE
from core.langchain_models import CachedModelFactory

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--swisscom_dataset", type=Path, 
                        help="Swisscom dataset JSON file.",
                        default="data/swisscom_dataset.json")
    parser.add_argument("--eval_qa_dataset", type=Path, 
                        help="Path to save evaluation QA dataset in JSON file.",
                        default="data/eval_qa_dataset.json")
    params = parser.parse_args()

    model_factory = CachedModelFactory(llm_cache_file="cache/llm.db",
                                    embeddings_cache_dir="cache/embeddings")
    
    model = model_factory.get_chat_model("gpt-4-0125-preview", temperature=0.7, top_p=0.1)
    with open(params.swisscom_dataset, "r") as f:
        dataset = json.load(f)

    print(len(dataset))
    random.seed(53)
    eval_sample = random.sample(dataset, 350)

    qa_eval_set = []
    json_parser = SimpleJsonOutputParser()

    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(QA_GEN_SYSTEM_PROMPT + "\n{community_prompt}\n" + QA_GEN_SYSTEM_FORMAT_PROMPT),
        HumanMessagePromptTemplate.from_template(QA_GEN_USER_PROMPT).format(**QA_GEN_EXAMPLE),
        AIMessage(content=json.dumps(QA_GEN_EXAMPLE["response"])),
        HumanMessagePromptTemplate.from_template(QA_GEN_USER_PROMPT)
    ])
    qa_gen_chain = prompt | model | SimpleJsonOutputParser()

    for sample in tqdm(eval_sample):
        community_prompt = QA_GEN_SYSTEM_COMMUNITY_PROMPT if sample["metadata"]["type"] == "community" else ""
        try:
            qa = qa_gen_chain.invoke({
                "community_prompt" : community_prompt,
                "context" : sample["text"]
            })
            if "question" not in qa or type(qa["question"]) != str:
                raise Exception("No valid question in JSON object.")
            if "answer" not in qa or type(qa["answer"]) != str:
                raise Exception("No valid answer in JSON object.")
        except Exception as e:
            # Skip in case of multiple failed retries
            print(f'Skipping {sample["metadata"]["id"]} due to `{e}`.')
            continue

        qa["relevant_id"] = sample["metadata"]["id"]
        qa_eval_set.append(qa)
    
    with open(params.eval_qa_dataset, "w") as f:
        json.dump(qa_eval_set, f, indent=1)