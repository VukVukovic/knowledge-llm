import os
import numpy as np
from pathlib import Path
from typing import List
from langchain.cache import SQLiteCache
from langchain.globals import set_llm_cache
from langchain.storage import LocalFileStore
from langchain.embeddings import CacheBackedEmbeddings
from langchain_core.stores import BaseStore
from langchain_core.embeddings import Embeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from .together_chat import ChatTogether
from langchain_mistralai.chat_models import ChatMistralAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.embeddings import CohereEmbeddings
from langchain_google_vertexai import ChatVertexAI
from langchain_google_vertexai import HarmCategory, HarmBlockThreshold
from langchain_anthropic import ChatAnthropic

safety_settings = {
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
}

def normalize(vector : List[float]) -> List[float]:
    return (vector / np.linalg.norm(vector)).tolist()

class FullyCacheBackedEmbeddings(CacheBackedEmbeddings):
    def __init__(
        self,
        underlying_embeddings: Embeddings,
        document_embedding_store: BaseStore[str, List[float]],
    ) -> None:
        super().__init__(underlying_embeddings, document_embedding_store)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [normalize(e) for e in super().embed_documents(texts)]
    
    def embed_query(self, text: str) -> List[float]:
        return normalize(self.embed_documents([text])[0])
    
    def pre_cache(self, texts: List[str]) -> None:
        self.embed_documents(texts)

class CachedModelFactory:
    def __init__(self, llm_cache_file : Path, embeddings_cache_dir : Path) -> None:
        set_llm_cache(SQLiteCache(database_path=llm_cache_file))
        self.embedding_store = LocalFileStore(embeddings_cache_dir)

    def get_embedding_model(self, model):
        if model in ["text-embedding-ada-002", "text-embedding-3-small",
                       "text-embedding-3-large"]:
            return FullyCacheBackedEmbeddings.from_bytes_store(
                OpenAIEmbeddings(model=model),
                self.embedding_store, namespace=model
            )

        if model in ["sentence-transformers/msmarco-bert-base-dot-v5", "BAAI/bge-large-en-v1.5", "WhereIsAI/UAE-Large-V1"]:
            return FullyCacheBackedEmbeddings.from_bytes_store(
                OpenAIEmbeddings(
                    model=model,
                    openai_api_base="https://api.together.xyz/v1",
                    openai_api_key=os.environ["TOGETHER_API_KEY"],
                    tiktoken_enabled=False,
                    embedding_ctx_length=500,
                    chunk_size=100
                ), 
                self.embedding_store, 
                namespace=model
            )
        
        if model in ["thenlper/gte-large"]:
            return FullyCacheBackedEmbeddings.from_bytes_store(
                OpenAIEmbeddings(
                    model=model,
                    openai_api_base="https://api.endpoints.anyscale.com/v1",
                    openai_api_key=os.environ["ANYSCALE_API_KEY"],
                    tiktoken_enabled=False,
                    embedding_ctx_length=500,
                    chunk_size=100
                ),
                self.embedding_store, 
                namespace=model
            )
        
        if model in ["mixedbread-ai/mxbai-embed-large-v1"]:
            return FullyCacheBackedEmbeddings.from_bytes_store(
                HuggingFaceEmbeddings(model_name=model, model_kwargs={"trust_remote_code":True}),
                self.embedding_store,
                namespace=model
            )
        
        if model in ["embed-english-light-v3.0", "embed-english-v3.0"]:
            return FullyCacheBackedEmbeddings.from_bytes_store(
                CohereEmbeddings(model=model),
                self.embedding_store,
                namespace=model
            )
        
        raise Exception(f"Embedding model `{model}` is not available.")
        
    def get_chat_model(self, model, **kwargs):
        if model in ["gpt-3.5-turbo-0125", "gpt-4-0125-preview"]:
            if "top_p" in kwargs:
                kwargs["model_kwargs"] = {"top_p" : kwargs["top_p"]}
                del kwargs["top_p"]
            return ChatOpenAI(model=model, **kwargs)
        
        if model in ["mistralai/Mistral-7B-Instruct-v0.2", "mistralai/Mixtral-8x7B-Instruct-v0.1",
                     "zero-one-ai/Yi-34B-Chat", "togethercomputer/llama-2-70b-chat", "Qwen/Qwen1.5-72B-Chat"]:
            return ChatTogether(model = model, **kwargs)
        
        if model in ["gemini-pro"]:
            return ChatVertexAI(model=model, convert_system_message_to_human=True,
                                          safety_settings=safety_settings, **kwargs)
        
        if model in ["mistral-small-latest", "mistral-medium-latest", "mistral-large-latest"]:
            return ChatMistralAI(model=model, **kwargs)
        
        if model in ["claude-3-opus-20240229", "claude-3-opus-20240229", "claude-3-haiku-20240307"]:
            return ChatAnthropic(model_name=model, **kwargs)
        
        raise Exception(f"LLM model `{model}` is not available.")
    
    @staticmethod
    def get_default_llm_params():
        return {
            "temperature": 0.7,
            "top_p": 0.1,
            "top_k": 40,
            "repetition_penalty": 1/0.85
        }