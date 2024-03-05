from typing import List
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables.base import RunnableSequence

class HydeRetriever(BaseRetriever):
    retriever: BaseRetriever
    hyde_chain: RunnableSequence

    @classmethod
    def from_hyde_chain(
        cls,
        retriever: BaseRetriever,
        hyde_chain: RunnableSequence
    ) -> "HydeRetriever":
        return cls(
            retriever=retriever,
            hyde_chain=hyde_chain
        )

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> List[Document]:
        hypothetical_doc = self.hyde_chain.invoke({"question": query})
        return self.retriever.get_relevant_documents(hypothetical_doc)