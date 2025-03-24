from flask import Flask, render_template, request, session, url_for, flash, redirect
import google.generativeai as genai
import fitz  # PyMuPDF for PDFs
import docx
from PIL import Image
import pytesseract
import uuid
import numpy as np
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

import pickle
app = Flask(__name__)
app.secret_key = "Hello123"

UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # Creates folder if not exist

# Load Embeddings
embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vector_store = None  # Initialize FAISS Index



# Configure Gemini API
genai.configure(api_key="AIzaSyBPezIOqsH6A2ZK3mh6KK1pfHwh3qu8pb4")
model = genai.GenerativeModel("gemini-1.5-flash")

# System Prompt for the chatbot
system_prompt = """You are Justice Chowdary, an AI assistant designed by Samuel for providing legal assistance in Indian laws and court-related matters. Your primary role is to assist users in accessing legal information, understanding laws, and offering insights into legal judgment predictions based on existing laws and precedents. 
You should provide precise, reliable, and well-referenced responses strictly within the domain of Indian law. 
If a user asks about non-legal topics, politely redirect them to legal matters. 
Do not provide personal legal advice but rather general legal information and references.
Maintain professionalism and clarity in all responses. If the given text is too large, give a brief explanation, don't dump entire text. 
Format the response to make it more readable."""

# Helper Functions
def extract_text_from_pdf(pdf_file):
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    text = "\n".join([page.get_text("text") for page in doc])
    return text

def extract_from_pdf(pdf_files):
    text = ""
    for file in pdf_files:
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(file_path)  # Save PDF
        pdf_reader = PdfReader(file_path)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def extract_text_from_docx(docx_file):
    doc = docx.Document(docx_file)
    text = "\n".join([para.text for para in doc.paragraphs])
    return text

def extract_text_from_image(image_file):
    image = Image.open(image_file)
    text = pytesseract.image_to_string(image)
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


def get_chatbot_response(prompt):
    full_prompt = system_prompt + "\n\nUser Query: " + prompt
    response = model.generate_content(full_prompt)
    return response.text

def summarize_text(text):
    summary_prompt = "Summarize the following legal document:\n\n" + text
    response = model.generate_content(summary_prompt)
    return response.text

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash",
                                   temperature=0.3)

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain

def get_user_response(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    response = chain(
        {"input_documents": docs, "question": user_question}
        , return_only_outputs=True)

    print(response)




# Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/models')
def models():
    return render_template('models.html')

@app.route('/chatbot', methods=["GET", "POST"])
def chatbot():
    # Reset session on reload (GET request) unless new_session or session_id is specified
    if request.method == "GET" and not request.args.get("new_session") and not request.args.get("session_id"):
        session.clear()

    # Initialize chat_sessions if not present
    if "chat_sessions" not in session:
        session["chat_sessions"] = {}

    # Handle new session creation
    if request.args.get("new_session"):
        new_id = str(uuid.uuid4())
        session["chat_sessions"][new_id] = {"title": "New Chat", "history": []}
        session["current_session"] = new_id
    # Handle session switching
    elif "session_id" in request.args:
        session_id = request.args.get("session_id")
        if session_id in session["chat_sessions"]:
            session["current_session"] = session_id
        else:
            flash("Invalid session ID", "error")
    # Initialize a default session if none exists
    elif "current_session" not in session or session["current_session"] not in session["chat_sessions"]:
        new_id = str(uuid.uuid4())
        session["chat_sessions"][new_id] = {"title": "New Chat", "history": []}
        session["current_session"] = new_id

    current_session_id = session["current_session"]
    current_chat_history = session["chat_sessions"][current_session_id]["history"]

    if request.method == "POST":
        if "chat_submit" in request.form:
            user_input = request.form.get("chat_input")
            if user_input:
                chat_response = get_chatbot_response(user_input)
                current_chat_history.append({"role": "user", "content": user_input})
                current_chat_history.append({"role": "assistant", "content": chat_response})
                # Set session title to first user message (trimmed to 30 chars)
                if len(current_chat_history) == 2:  # First user and assistant messages
                    title = user_input[:30] + "..." if len(user_input) > 30 else user_input
                    session["chat_sessions"][current_session_id]["title"] = title
        elif "upload_submit" in request.form:
            file = request.files.get("upload_file")
            if file:
                filename = file.filename
                if file.mimetype == "application/pdf":
                    extracted_text = extract_text_from_pdf(file)
                elif file.mimetype == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                    extracted_text = extract_text_from_docx(file)
                else:
                    extracted_text = extract_text_from_image(file)
                session["extracted_text"] = extracted_text
                current_chat_history.append({"role": "user", "content": f"User attached a file: {filename}"})
                summary = summarize_text(extracted_text)
                current_chat_history.append({"role": "assistant", "content": f"Here's a summary of the document:\n\n{summary}"})

        # Update the session
        session["chat_sessions"][current_session_id]["history"] = current_chat_history

    sessions = session["chat_sessions"]
    chat_history = current_chat_history
    return render_template("chatbot.html", chat_history=chat_history, sessions=sessions, current_session_id=current_session_id)

@app.route('/document',methods=["GET", "POST"])
def document():
    if request.method == "POST":
        # Handle PDF Upload
        if "pdf_files" in request.files:
            pdf_files = request.files.getlist("pdf_files")
            extracted_text = extract_from_pdf(pdf_files)
            text_chunks = get_text_chunks(extracted_text)
            get_vector_store(text_chunks)
            flash("PDFs processed successfully!", "success")

        if "user_question" in request.form:
            user_question = request.form["user_question"]
            answer = get_user_response(user_question)
            return render_template("document.html", answer=answer)

    return render_template('document.html')


if __name__ == "__main__":
    app.run(debug=True)
