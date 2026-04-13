from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
from transformers import pipeline

# Sample dataset
docs = [
    Document(page_content="Python is a programming language."),
    Document(page_content="Machine learning is a subset of AI."),
    Document(page_content="RAG combines retrieval with generation.")
]

# Embeddings
embedding_model = HuggingFaceEmbeddings()

# Vector store
db = FAISS.from_documents(docs, embedding_model)
retriever = db.as_retriever()

# LLM
pipe = pipeline("text-generation", model="gpt2")
llm = HuggingFacePipeline(pipeline=pipe)

# RAG pipeline
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# Chat loop
while True:
    query = input("You: ")
    if query.lower() == "exit":
        break
    response = qa_chain.run(query)
    print("Bot:", response)