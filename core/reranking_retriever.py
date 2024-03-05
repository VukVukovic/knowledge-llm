from typing import List
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.retrievers import BaseRetriever
from langchain.chains.llm import LLMChain
from langchain_core.pydantic_v1 import Field
from typing import Any

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

class RerankingRetriever(BaseRetriever):
    retriever: BaseRetriever
    k: int
    tokenizer: Any = Field(default=None, exclude=True)  #: :meta private:
    model: Any = Field(default=None, exclude=True)  #: :meta private:
    device: Any = Field(default=None, exclude=True) #: :meta private:

    @classmethod
    def from_hf_model(
        cls,
        retriever: BaseRetriever,
        model_name: str,
        k: int
    ) -> "RerankingRetriever":
        device = "cpu"
        if torch.backends.mps.is_available():
            device = "mps"
        if torch.cuda.is_available():
            device = "cuda"
        device = torch.device(device)

        return cls(
            retriever=retriever,
            k=k,
            tokenizer=AutoTokenizer.from_pretrained(model_name),
            model=AutoModelForSequenceClassification.from_pretrained(model_name).to(device),
            device=device,
        )
    
    def _score(self, query: str, documents: List[Document]) -> List[float]:
        pairs = [(query, d.page_content) for d in documents]
        with torch.no_grad():
            inputs = self.tokenizer(pairs, padding=True, truncation=True, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            scores = self.model(**inputs, return_dict=True).logits.view(-1,).float().cpu()
        return scores.tolist()

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> List[Document]:
        relevant_docs = self.retriever.get_relevant_documents(query)
        scores = self._score(query, relevant_docs)
        reranked_docs = [d for _, d in sorted(zip(scores, relevant_docs), key=lambda p: p[0], reverse=True)]
        if len(reranked_docs) >= self.k:
            reranked_docs = reranked_docs[:self.k]
        return reranked_docs