import os
from fastapi import APIRouter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

router = APIRouter()

os.makedirs("memory_data", exist_ok=True)

def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return splitter.split_text(text)

def store_to_vector_db(text, username):
    chunks = get_text_chunks(text)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    db_path = f"memory_data/{username}"
    if os.path.exists(db_path):
        vector_store = FAISS.load_local(db_path, embeddings, allow_dangerous_deserialization=True)
        vector_store.add_texts(chunks)
    else:
        vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    vector_store.save_local(db_path)

def query_memory(query, username):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    db_path = f"memory_data/{username}"
    db = FAISS.load_local(db_path, embeddings, allow_dangerous_deserialization=True)
    docs = db.similarity_search(query)

    prompt = PromptTemplate(
        template="""
        Answer using context below. If answer is not found, say:
        "Sorry, not found in memory."

        Context:
        {context}

        Question:
        {question}

        Answer:
        """,
        input_variables=["context", "question"]
    )

    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.3)
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    result = chain({"input_documents": docs, "question": query}, return_only_outputs=True)
    return result["output_text"]

@router.post("/memorize/{username}")
def memorize_text(username: str, body: dict):
    text = body.get("text", "")
    if not text.strip():
        return {"error": "Empty text"}
    store_to_vector_db(text, username)
    return {"message": "Text memorized."}

@router.post("/query")
def ask_question(request: dict):
    username = request["username"]
    query = request["query"]
    answer = query_memory(query, username)
    return {"answer": answer}
