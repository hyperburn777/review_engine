from langchain_community.chat_models import ChatOllama
from langchain_community.vectorstores import Chroma
from langchain_core.embeddings import Embeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
import json


class STEmbeddings(Embeddings):
    def __init__(self, model):
        self.model = model

    def embed_documents(self, texts):
        return self.model.encode(texts).tolist()

    def embed_query(self, text):
        return self.model.encode([text])[0].tolist()


class RAG:

    def __init__(self, model, default_meta, rag_model="llama3"):
        self.model = model
        self.emb = STEmbeddings(model)
        self.llm = ChatOllama(
            model=rag_model,
            temperature=0.2,
            num_ctx=4096,
        )
        self.prompt = ChatPromptTemplate.from_template(
            """Use the following pieces of context to answer the question at the end.
            If you aren't sure of the answer, say you don't know instead of guessing.

            {context}

            Question: {question}
            Helpful Answer:"""
        )
        self.change_product(default_meta)

        self.rag_chain = (
            RunnableParallel({"context": self.retriever, "question": RunnablePassthrough()})
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

    def change_product(self, new_meta):
        product_text = json.dumps(new_meta, indent=2)
        store = Chroma.from_texts([product_text], embedding=self.emb)
        self.retriever = store.as_retriever()
    
    def generate_answer(self, question):
        if self.rag_chain is None:
            return "You should choose a product to ask further questions"
        return self.rag_chain.invoke(question)
