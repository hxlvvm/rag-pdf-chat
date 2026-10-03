"""Retrieval-augmented answering: retrieve the most relevant chunks, then ask the LLM to answer only from them."""
from dataclasses import dataclass

from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You answer questions using ONLY the context below. If the answer is not in the context, say "
               "you don't know. Cite sources as [source p.page].\n\nContext:\n{context}"),
    ("human", "{question}"),
])


@dataclass
class Answer:
    answer: str
    sources: list[dict]


def format_context(docs) -> str:
    return "\n\n".join(f"[{d.metadata.get('source')} p.{d.metadata.get('page')}] {d.page_content}" for d in docs)


class RagPipeline:
    def __init__(self, store, llm: BaseChatModel, top_k: int = 4):
        self.retriever = store.as_retriever(search_kwargs={"k": top_k})
        self.chain = PROMPT | llm | StrOutputParser()

    def ask(self, question: str) -> Answer:
        docs = self.retriever.invoke(question)
        text = self.chain.invoke({"context": format_context(docs), "question": question})
        return Answer(answer=text, sources=[{"source": d.metadata.get("source"), "page": d.metadata.get("page")}
                                            for d in docs])


def default_llm(model: str) -> BaseChatModel:
    from langchain_groq import ChatGroq  # needs GROQ_API_KEY (free tier at console.groq.com)

    return ChatGroq(model=model, temperature=0)
