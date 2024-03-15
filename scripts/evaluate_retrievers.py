import sys
import argparse
import json
from pathlib import Path
from typing import List
from tqdm import tqdm

from langchain.chains import LLMChain
from langchain_core.output_parsers import BaseOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain.vectorstores.utils import DistanceStrategy
from langchain.schema import Document

from langchain.prompts import (
        ChatPromptTemplate,
        HumanMessagePromptTemplate,
        SystemMessagePromptTemplate
    )

# Support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from core.langchain_models import CachedModelFactory
from core.prompt_bank import MULTI_QUERY_SYSTEM, MULTI_QUERY_USER, HYDE_SYSTEM, HYDE_USER
from core.multi_query import MultiQueryRetriever
from core.reranking_retriever import RerankingRetriever
from core.hyde_retriever import HydeRetriever

class LineListOutputParser(BaseOutputParser[List[str]]):
    def parse(self, text: str) -> List[str]:
        return text.strip().split("\n")
    
def get_langchain_documents(swisscom_dataset):
    return [
        Document(
            page_content = d["text"],
            metadata = d["metadata"],
        ) for d in swisscom_dataset
    ]

def evaluate_retriever(retriever, eval_qa_dataset):
    hits = 0
    for q in tqdm(eval_qa_dataset):
        docs = retriever.get_relevant_documents(q["question"])
        retrieved_ids = set([d.metadata["id"] for d in docs])
        if q["relevant_id"] in retrieved_ids:
            hits += 1
    return hits/len(eval_qa_dataset)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--swisscom_dataset", type=Path, 
                        help="Swisscom dataset JSON file.",
                        default="data/swisscom_dataset.json")
    parser.add_argument("--eval_qa_dataset", type=Path, 
                        help="Evaluation QA dataset JSON file.",
                        default="data/eval_qa_dataset.json")
    params = parser.parse_args()

    # Initialize model factory with cache for both embeddings and llms
    model_factory = CachedModelFactory(llm_cache_file="cache/llm.db",
                                    embeddings_cache_dir="cache/embeddings")
    
    # Load documents and evaluation dataset
    with open(params.swisscom_dataset, "r") as f:
        swisscom_dataset = json.load(f)
    with open(params.eval_qa_dataset, "r") as f:
        eval_qa_dataset = json.load(f)
    documents = get_langchain_documents(swisscom_dataset)

    TOP_K = 3

    # BM25
    bm25_retriever = BM25Retriever.from_documents(documents, k=TOP_K)

    # Embeddings
    embedding_model = model_factory.get_embedding_model(model="mixedbread-ai/mxbai-embed-large-v1")
    embedding_vector_store = FAISS.from_documents(documents, embedding_model, 
                                                distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT)
    embeddings_retriever = embedding_vector_store.as_retriever(search_kwargs={"k": TOP_K})
    
    # Reranking BAAI/bge-reranker-large
    embeddings_retriever_10 = embedding_vector_store.as_retriever(search_kwargs={"k": 10})
    #reranking_retriever_bge = RerankingRetriever.from_hf_model(retriever=embeddings_retriever_10, 
    #                                                       model_name="BAAI/bge-reranker-large", k=TOP_K)
    #reranking_retriever_mxbai = RerankingRetriever.from_hf_model(retriever=embeddings_retriever_10,
    #                                                        model_name="mixedbread-ai/mxbai-rerank-base-v1", k=TOP_K)
    
    # Fusion
    fusion_retriever = EnsembleRetriever(
        retrievers=[embeddings_retriever, bm25_retriever], weights=[0.6, 0.4]
    )
    
    # Multi-query                            
    llm_model = model_factory.get_chat_model("mistralai/Mistral-7B-Instruct-v0.2", max_tokens=512, 
                                             **CachedModelFactory.get_default_llm_params())
    multi_query_prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(MULTI_QUERY_SYSTEM),
        HumanMessagePromptTemplate.from_template(MULTI_QUERY_USER)
    ])
    multi_query_chain = LLMChain(llm=llm_model, prompt=multi_query_prompt, output_parser=LineListOutputParser())
    multi_query_retriever = MultiQueryRetriever.from_llm(retriever=embeddings_retriever, 
                                                         query_chain=multi_query_chain, include_original=True)
    
    # HyDE
    hyde_prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(HYDE_SYSTEM),
        HumanMessagePromptTemplate.from_template(HYDE_USER)
    ])
    hyde_chain = hyde_prompt | llm_model | StrOutputParser()
    hyde_retriever = HydeRetriever.from_hyde_chain(retriever=embeddings_retriever, hyde_chain=hyde_chain)

    print(f"BM25: {evaluate_retriever(bm25_retriever, eval_qa_dataset)}")
    print(f"Embeddings: {evaluate_retriever(embeddings_retriever, eval_qa_dataset)}")
    print(f"Fusion: {evaluate_retriever(fusion_retriever, eval_qa_dataset)}")
    print(f"Multi-query: {evaluate_retriever(multi_query_retriever, eval_qa_dataset)}")
    print(f"HyDE: {evaluate_retriever(hyde_retriever, eval_qa_dataset)}")
    #print(f"Reranking mxbai: {evaluate_retriever(reranking_retriever_mxbai, eval_qa_dataset)}")
    #Reranking mxbai: 0.7971428571428572
    #print(f"Reranking bge: {evaluate_retriever(reranking_retriever_bge, eval_qa_dataset)}")
    #Reranking bge: 0.8114285714285714