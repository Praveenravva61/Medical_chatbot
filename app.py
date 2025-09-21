from flask import Flask, render_template, jsonify, request
from langchain_perplexity import ChatPerplexity
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import system_prompt
import os
from env import PERPLEXITY_API_KEY, PINECONE_API_KEY


load_dotenv()



app = Flask(__name__)

embeddings = download_hugging_face_embeddings()

index_name = "medical-chatbot"

docsearch = PineconeVectorStore.from_existing_index(
    index_name= index_name,
    embedding=embeddings
)

retriever = docsearch.as_retriever(
    search_type = "similarity",
    search_kwargs ={"k":3}

)

model = ChatPerplexity(model= 'sonar', api_key=PERPLEXITY_API_KEY)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human","{input}")
    ]
)

question_answer_chain = create_stuff_documents_chain(model, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

@app.route("/")   # <-- fixed
def index():
    return render_template("chat.html")


@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form['msg']
    input = msg
    print(input)
    response= rag_chain.invoke({"input":msg})
    
    return str(response['answer'])






if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
    
    

