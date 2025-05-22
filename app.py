import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Load API key
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


# Text chunking
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks


# Store to FAISS vector DB
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


# Prompt + Chain
def get_conversational_chain():
    prompt_template = """
    You are an intelligent assistant. Answer the question using only the information provided in the context.

    Be clear, complete, and concise. Respond in full sentences. If the answer is not found in the context, say:
    "Sorry, I couldn't find that information in the memory."

    Context:
    {context}

    Question:
    {question}

    Answer:
    """

    model = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash-latest",
        temperature=0.3
    )

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain


# Handle queries
def user_input(user_question, user_id):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    db_path = f"memory_data/{user_id}"
    new_db = FAISS.load_local(db_path, embeddings, allow_dangerous_deserialization=True)
    if not new_db:
        st.warning("⚠️ No memory found for this user ID.")
        return
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)

    st.write("🤖 Reply:", response["output_text"])


# Text input from user
def get_text_input():
    if "memory_input" not in st.session_state:
        st.session_state.memory_input = ""

    return st.text_area(
        "🧠 Enter any text you'd like the AI to memorize:",
        height=200,
        key="memory_input")
     


# Save memory
def memorize_text(input_text, user_id):
    chunks = get_text_chunks(input_text)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    db_path = f"memory_data/{user_id}"

    if os.path.exists(db_path):
        vector_store = FAISS.load_local(db_path, embeddings, allow_dangerous_deserialization=True)
        vector_store.add_texts(chunks)
    else:
        vector_store = FAISS.from_texts(chunks, embedding=embeddings)

    vector_store.save_local(f"memory_data/{user_id}")
    st.session_state["memory_input"] = ""  # Clear the text box after saving
    st.session_state["user_id"] = user_id  # Store user ID in session state


def clear_input():
    st.session_state.memory_input = ""

# Streamlit UI
def main():
    st.set_page_config(page_title="AI Text Memory Assistant")
    st.header("💡 AI Text Memorizer with Future Querying")

    # User identity input
    user_id = st.text_input("🔐 Enter your email or username:", key="user_id")

    if not user_id.strip():
        st.warning("⚠️ Please enter your user ID to begin.")
        return

    # Input text to memorize
    input_text = get_text_input()
    if st.button("💾 Memorize this text", on_click=clear_input):
        if input_text.strip():
            with st.spinner("Memorizing..."):
                memorize_text(input_text, user_id)
                st.success("✅ Memory stored successfully!")
                st.session_state["memory_input"] = ""  # Clear the text box
        else:
            st.warning("⚠️ Please enter some text before memorizing.")

    # Ask a question
    user_question = st.text_input("❓ Ask a question from the memorized content")
    if user_question.strip():
        user_input(user_question, user_id)


if __name__ == "__main__":
    main()
