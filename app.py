from flask import Flask, render_template, request, session, redirect, url_for,flash
import google.generativeai as genai
import fitz  # PyMuPDF for PDFs
import docx
from PIL import Image
import pytesseract
import uuid
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# from langchain.vectorstores import FAISS
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain.chains.question_answering import load_qa_chain
# from langchain.prompts import PromptTemplate

app = Flask(__name__)
app.secret_key = "Hello123"  # Replace with a secure key in production

# Configure Gemini API
genai.configure(api_key="AIzaSyBPezIOqsH6A2ZK3mh6KK1pfHwh3qu8pb4")
model = genai.GenerativeModel("gemini-1.5-flash")

# System Prompt for the chatbot
system_prompt = """You are Justice Chowdary, an AI assistant designed by Samuel for providing legal assistance in Indian laws and court-related matters. Your primary role is to assist users in accessing legal information, understanding laws, and offering insights into legal judgment predictions based on existing laws and precedents. 
You should provide precise, reliable, and well-referenced responses strictly within the domain of Indian law. 
If a user asks about non-legal topics, politely redirect them to legal matters. 
Do not provide personal legal advice but rather general legal information and references.
 Maintain professionalism and clarity in all responses. If the given text is too large, give a brief explanation, don't dump entire text. 
 format the response, to make it more readable """


# -------------------- Helper Functions --------------------
def extract_text_from_pdf(pdf_file):
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    text = "\n".join([page.get_text("text") for page in doc])
    return text


def extract_text_from_docx(docx_file):
    doc = docx.Document(docx_file)
    text = "\n".join([para.text for para in doc.paragraphs])
    return text


def extract_text_from_image(image_file):
    image = Image.open(image_file)
    text = pytesseract.image_to_string(image)
    return text


def get_chatbot_response(prompt):
    full_prompt = system_prompt + "\n\nUser Query: " + prompt
    response = model.generate_content(full_prompt)
    return response.text


def summarize_text(text):
    summary_prompt = "Summarize the following legal document:\n\n" + text
    response = model.generate_content(summary_prompt)
    return response.text
# #
# def get_pdf_text(pdf_files):
#     """Extract text from a list of PDF files."""
#     text = ""
#     for pdf in pdf_files:
#         pdf_reader = PdfReader(pdf)
#         for page in pdf_reader.pages:
#             page_text = page.extract_text()
#             if page_text:
#                 text += page_text + "\n"
#     return text
#
# def get_text_chunks(text):
#     """Split the text into manageable chunks."""
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
#     chunks = text_splitter.split_text(text)
#     return chunks
#
# def get_vector_store(text_chunks):
#     """Create a FAISS vector store from text chunks and save it locally."""
#     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
#     vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
#     vector_store.save_local("faiss_index")
#     return vector_store
#
# def get_conversational_chain():
#     """Create a question-answering chain using a custom prompt."""
#     prompt_template = """
#     Answer the question as detailed as possible from the provided context. Make sure to provide all the details.
#     If the answer is not in the provided context, just say, "answer is not available in the context". Do not provide a wrong answer.
#
#     Context:
#     {context}
#
#     Question:
#     {question}
#
#     Answer:
#     """
#     model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
#     prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
#     chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
#     return chain
#
# def answer_question(user_question):
#     """Load the FAISS index, perform similarity search, and run the QA chain."""
#     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
#     vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
#     docs = vector_store.similarity_search(user_question)
#     chain = get_conversational_chain()
#     response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
#     return response["output_text"]

# ------------------ Routes ------------------

@app.route("/document", methods=["GET", "POST"])
def document():
    answer = None
    if request.method == "POST":
        # If PDF is uploaded
        if "upload_submit" in request.form:
            pdf_files = request.files.getlist("pdf_files")
            if pdf_files and any(pdf.filename for pdf in pdf_files):
                raw_text = ""
                for pdf in pdf_files:
                    raw_text += extract_text_from_pdf(pdf) + "\n"
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                flash("PDF processing complete!", "success")
            else:
                flash("Please upload at least one PDF file.", "warning")
            return redirect(url_for("document"))
        # If question is asked
        elif "ask_submit" in request.form:
            user_question = request.form.get("user_question")
            if user_question:
                answer = answer_question(user_question)
            else:
                flash("Please enter a question.", "warning")
    return render_template("document.html", answer=answer)

# -------------------- Routes --------------------
@app.route('/')
def home():
    return render_template('index.html')


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/models')
def models():
    return render_template('models.html')


# Chatbot route
@app.route('/chatbot', methods=["GET", "POST"])
def chatbot():
    if request.args.get("new_session"):
        session.pop("chat_history", None)
        session.pop("extracted_text", None)


    if "chat_history" not in session:
        session["chat_history"] = []

    if request.method == "POST":
        # Handle chat message submission
        if "chat_submit" in request.form:
            user_input = request.form.get("chat_input")
            chat_response = get_chatbot_response(user_input)
            session["chat_history"].append({"role": "user", "content": user_input})
            session["chat_history"].append({"role": "assistant", "content": chat_response})
        # Handle file upload and summary
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
                # Add upload notification and summary to chat history
                session["chat_history"].append({"role": "user", "content": f"User attached a file: {filename}"})
                summary = summarize_text(extracted_text)
                session["chat_history"].append({"role": "assistant", "content": f"Here's a summary of the document:\n\n{summary}"})

    chat_history = session.get("chat_history", [])
    sessions = {}  # Add session management logic if needed
    return render_template("chatbot.html", chat_history=chat_history, sessions=sessions)

# @app.route('/document')
# def document():
#     return render_template('document.html')

if __name__ == "__main__":
    app.run(debug=True)
