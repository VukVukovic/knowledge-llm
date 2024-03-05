from typing import List
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.retrievers import BaseRetriever
from langchain.chains.llm import LLMChain

class MultiQueryRetriever(BaseRetriever):
    retriever: BaseRetriever
    query_chain: LLMChain
    include_original: bool = False

    @classmethod
    def from_llm(
        cls,
        retriever: BaseRetriever,
        query_chain: LLMChain,
        include_original: bool = False,
    ) -> "MultiQueryRetriever":
        return cls(
            retriever=retriever,
            query_chain=query_chain,
            include_original=include_original,
        )

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> List[Document]:
        queries = self.generate_queries(query, run_manager)
        if self.include_original:
            queries.append(query)
        documents = self.retrieve_documents(queries, run_manager)
        return self.unique_union(documents)

    def generate_queries(self, question: str, run_manager: CallbackManagerForRetrieverRun) -> List[str]:
        response = self.query_chain.invoke({"question": question}, run_manager=run_manager)
        return response["text"]
    
    def retrieve_documents(
        self, queries: List[str], run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        documents = []
        for query in queries:
            docs = self.retriever.get_relevant_documents(
                query, callbacks=run_manager.get_child()
            )
            documents.extend(docs)
        return documents

    def unique_union(self, documents: List[Document]) -> List[Document]:
        return [doc for i, doc in enumerate(documents) if doc not in documents[:i]]