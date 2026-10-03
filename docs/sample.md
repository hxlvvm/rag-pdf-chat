# Sample knowledge base

Retrieval-augmented generation (RAG) answers questions by first retrieving relevant passages from a
document collection and then asking a language model to answer using only those passages.
A vector database such as Chroma stores an embedding for every chunk of text, so passages with a
similar meaning can be found even when they use different words.
