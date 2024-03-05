import os
import json
import argparse
import pandas as pd
from pathlib import Path
import uuid
import random
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter

rd = random.Random()
rd.seed(47)
uuid.UUID(int=rd.getrandbits(128))

def len_in_words(text: str) -> int:
    return len(re.findall(r'\b\w+\b', text))

# Chunk size taken from Plato.
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=75,
    length_function=len_in_words
)

def clean_website(dataset):
    print("=== Webpage dataset")
    print(f"Original dataset size: {len(dataset)}")

    en_dataset = list(filter(lambda d : d["metadata"]["language"]=="en", dataset))
    print(f"English dataset size: {len(en_dataset)}")

    # Webpage dataset is already chunked (500 words, 75 word overlap).
    unique_en_dataset = []
    unique_texts = set()
    for datapoint in en_dataset:
        if not datapoint["text"] in unique_texts:
            unique_texts.add(datapoint["text"])
            new_datapoint = datapoint.copy()
            del new_datapoint["vector_field"]
            new_datapoint["metadata"]["type"] = "webpage"
            unique_en_dataset.append(new_datapoint)

    print(f"English dataset without duplicates size: {len(unique_en_dataset)}")
    return unique_en_dataset

def clean_community(dataset):
    print("=== Community dataset")
    print(f"Original dataset size: {len(dataset)}")

    en_dataset = dataset[dataset["language"]=="EN"]
    print(f"English dataset size: {len(en_dataset)}")

    unique_texts = set()
    unique_en_dataset = []

    texts = []
    for _, row in en_dataset.iterrows():
        if type(row["question"]) != str or type(row["answer"]) != str:
            continue

        text = row["question"] + "\n" + row["answer"]
        if text in unique_texts:
            continue #skip duplicates

        texts.append(text)
        
        # Chunk documents (QAs)
        start_index = 0
        for chunk in text_splitter.split_text(text):
            doc = {
                "text" : chunk,
                "metadata" : {
                    "type": "community",
                    "language": row["language"].lower(),
                    "source": row["url"],
                    "solved": row["solved"],
                    "start_index": start_index
                }
            }
            start_index += len(chunk)
            unique_en_dataset.append(doc)
    
    print(f"English dataset without duplicates size (chunked): {len(unique_en_dataset)}")
    
    return unique_en_dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--website_dataset", type=Path, 
                        help="Swisscom website dataset JSON file (from Plato).",
                        default="data/webpage_dataset.json")
    
    parser.add_argument("--community_dataset", type=Path, 
                        help="Swisscom community dataset CSV file.",
                        default="data/community_dataset.csv")
    
    parser.add_argument("--swisscom_dataset", type=Path, 
                        help="Resulting Swisscom dataset JSON file to save.",
                        default="data/swisscom_dataset.json")
    
    params = parser.parse_args()

    with open(params.website_dataset, "r") as f:
        website_dataset = json.load(f)
    
    community_dataset = pd.read_csv(params.community_dataset)

    website_dataset = clean_website(website_dataset)
    print()
    community_dataset = clean_community(community_dataset)
    print()

    print("=== Final dataset")
    dataset = website_dataset + community_dataset
    for datapoint in dataset:
        datapoint["metadata"]["id"] = str(uuid.uuid4())

    print(f"Full dataset size: {len(dataset)}")
    print(f"Max document chunk size in words: {max([len_in_words(d['text']) for d in dataset])}")
    
    with open(params.swisscom_dataset, "w") as f:
        json.dump(dataset, f, indent=1)
