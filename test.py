from flask import Flask, render_template, jsonify, request, session
from langchain_perplexity import ChatPerplexity
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
from src.prompt import system_prompt
import asyncio
import os
from env import PERPLEXITY_API_KEY

load_dotenv()

app = Flask(__name__)
app.secret_key = os.urandom(24)  # For session encryption

embeddings = download_hugging_face_embeddings()
index_name = "medical-chatbot"

docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

retriever = docsearch.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)

model = ChatPerplexity(model='sonar', api_key=PERPLEXITY_API_KEY)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}")
    ]
)

question_answer_chain = create_stuff_documents_chain(model, prompt)

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Create a retrieval chain that uses conversation memory for context
rag_chain = create_retrieval_chain(retriever, question_answer_chain, memory=memory)


@app.route("/")
def index():
    return render_template("chat.html")


@app.route("/get", methods=["POST"])
async def chat():
    msg = request.form['msg']
    if "chat_history" not in session:
        session["chat_history"] = []

    # Add current user message to session history
    session["chat_history"].append({"role": "user", "content": msg})

    # Update memory with session chat history
    memory.chat_memory.messages = session["chat_history"]

    # Run the RAG chain asynchronously
    response = await asyncio.to_thread(rag_chain.invoke, {"input": msg})

    answer = response["answer"]

    # Add bot response to session history
    session["chat_history"].append({"role": "assistant", "content": answer})

    session.modified = True

    return jsonify({"answer": answer})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
