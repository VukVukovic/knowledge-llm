import argparse
import os
import sys
import json
from pathlib import Path
from retry import retry
from tqdm import tqdm
from langchain.output_parsers.json import SimpleJsonOutputParser

# Support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from core.dataset import SwisscomDataset
from core.cached_model import CachedModelClient, ModelClient
from core.prompt_bank import QA_GEN_SYSTEM_PROMPT, QA_GEN_SYSTEM_COMMUNITY_PROMPT, QA_GEN_USER_PROMPT, QA_GEN_USER_PROMPT, QA_GEN_SYSTEM_FORMAT_PROMPT

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--swisscom_dataset", type=Path, 
                        help="Swisscom dataset JSON file.",
                        default="data/swisscom_dataset.json")
    parser.add_argument("--eval_qa_dataset", type=Path, 
                        help="Path to save evaluation QA dataset in JSON file.",
                        default="data/eval_qa_dataset.json")
    params = parser.parse_args()
    
    client = ModelClient(openai_key=os.environ["OPENAI_API_KEY"])
    cached_client = CachedModelClient(client, "cache/model_cache.pcikle")

    dataset = SwisscomDataset(params.swisscom_dataset)
    eval_sample = dataset.get_eval_sample(size=350)

    qa_eval_set = []
    json_parser = SimpleJsonOutputParser()
    for sample in tqdm(eval_sample):
        system_prompt = QA_GEN_SYSTEM_PROMPT
        if sample["metadata"]["type"] == "community":
            system_prompt += QA_GEN_SYSTEM_COMMUNITY_PROMPT
        system_prompt += QA_GEN_SYSTEM_FORMAT_PROMPT
        user_prompt = QA_GEN_USER_PROMPT.format(context = sample["text"])

        try:
            qa_sample_text = cached_client.get_completion(
                        system_prompt, user_prompt, "gpt-4-0125-preview",
                        temperature=0.7, response_format={ "type": "json_object" })
            qa_sample = json_parser.parse(qa_sample_text)
            if "question" not in qa_sample or type(qa_sample["question"]) != str:
                raise Exception("No valid question in JSON object.")
            if "answer" not in qa_sample or type(qa_sample["answer"]) != str:
                raise Exception("No valid answer in JSON object.")
        except Exception as e:
            # Skip in case of multiple failed retries
            print(f'Skipping {sample["metadata"]["id"]} due to `{e}`.')
            continue

        qa_sample["relevant_id"] = sample["metadata"]["id"]
        qa_eval_set.append(qa_sample)
    
    with open(params.eval_qa_dataset, "w") as f:
        json.dump(qa_eval_set, f, indent=1)