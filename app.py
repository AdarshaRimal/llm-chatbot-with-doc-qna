import os
import tempfile
import streamlit as st
from streamlit_chat import message
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import Tool
from langchain.memory import ConversationBufferWindowMemory
from chatbot.document_loader import load_document, split_documents
from chatbot.vector_store import VectorStoreManager
from chatbot.qa_chain import build_qa_chain, get_answer
from chatbot.form_handler import AppointmentForm
from chatbot.agent import build_agent

load_dotenv()

st.set_page_config(page_title="PalmMind AI Assistant", page_icon="🤖", layout="centered")

class SessionState:
    def __init__(self):
        self.chat_history = []
        self.uploaded_file = None
        self.vector_store = None
        self.qa_chain = None
        self.agent = None
        self.form = AppointmentForm()
        self.processing_lock = False
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0.3,
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
        # Add conversation memory here
        self.memory = ConversationBufferWindowMemory(
            memory_key="chat_history",
            k=10,  # Keep last 10 messages in memory
            return_messages=True
        )

def initialize_session():
    if "state" not in st.session_state:
        st.session_state.state = SessionState()
    if not st.session_state.state.agent:
        st.session_state.state.agent = build_base_agent()

def build_base_agent():
    tools = [
        Tool(
            name="BookAppointment",
            func=st.session_state.state.form.agent_trigger,
            description="Schedule appointments or calls"
        )
    ]
    return build_agent(
        llm=st.session_state.state.llm,
        tools=tools,
        system_message="You are a helpful assistant. Ask for documents if needed.",
        memory=st.session_state.state.memory
    )

def build_document_agent(retriever):
    doc_qa_chain = build_qa_chain(retriever)

    def document_qa_func(query: str) -> str:
        result = get_answer(doc_qa_chain, query)
        answer = result["answer"]

        # Update conversation memory manually to remember document Q&A context
        st.session_state.state.memory.chat_memory.add_user_message(query)
        st.session_state.state.memory.chat_memory.add_ai_message(answer)

        return answer

    tools = [
        Tool(
            name="DocumentQA",
            func=document_qa_func,
            description="Answer questions about uploaded document"
        ),
        Tool(
            name="BookAppointment",
            func=st.session_state.state.form.agent_trigger,
            description="Schedule appointments or calls"
        )
    ]

    return build_agent(
        llm=st.session_state.state.llm,
        tools=tools,
        system_message=(
            "You are a helpful assistant with access to two tools:\n"
            "1. DocumentQA - Use this tool to answer any questions about the uploaded document.\n"
            "2. BookAppointment - Use this tool to schedule appointments.\n\n"
            "When you want to use a tool, respond exactly in the following format:\n"
            "Thought: <your thought>\n"
            "Action: <tool name>\n"
            "Action Input: <input for the tool as plain text>\n\n"
            "If you know the final answer without using a tool, respond with:\n"
            "Final Answer: <your answer>\n\n"
            "Do not use JSON, do not add code blocks or markdown formatting.\n"
            "Begin!\n"
        ),
        memory=st.session_state.state.memory
    )

def process_uploaded_file(uploaded_file):
    try:
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(uploaded_file.getvalue())
            temp_path = temp_file.name

        docs = load_document(temp_path)
        chunks = split_documents(docs)
        vector_mgr = VectorStoreManager()
        vector_store = vector_mgr.create_store(chunks)

        st.session_state.state.vector_store = vector_store
        st.session_state.state.agent = build_document_agent(vector_store.as_retriever())
        os.unlink(temp_path)
        st.success("✅ Document processed successfully!")
    except Exception as e:
        import traceback
        st.error(f"❌ Error processing document: {str(e)}")
        st.text(traceback.format_exc())

def handle_user_input(user_input: str):
    state = st.session_state.state

    if state.processing_lock or not user_input.strip():
        return

    state.processing_lock = True
    try:
        if state.form.active or state.form.should_start(user_input):
            # Your form logic here if using form-based interaction
            pass
        else:
            doc_keywords = ["document", "summary", "summarize", "extract", "fact", "pdf", "file"]
            if any(kw in user_input.lower() for kw in doc_keywords):
                if state.vector_store is None:
                    response_text = "Please provide the document you would like me to summarize."
                    state.chat_history.extend([
                        {"role": "user", "content": user_input},
                        {"role": "assistant", "content": response_text}
                    ])
                    return

            if state.agent:
                response = state.agent.invoke({"input": user_input})
                clean_response = response["output"].split("SOURCES:")[0].strip()
                state.chat_history.extend([
                    {"role": "user", "content": user_input},
                    {"role": "assistant", "content": clean_response}
                ])

    except Exception as e:
        error_msg = f"⚠️ System error: {str(e)}"
        state.chat_history.append({"role": "assistant", "content": error_msg})
    finally:
        state.processing_lock = False

def reset_chat():
    st.session_state.state.chat_history = []
    st.session_state.state.form = AppointmentForm()
    st.session_state.state.vector_store = None
    st.session_state.state.agent = build_base_agent()
    st.session_state.state.uploaded_file = None
    # Reset memory as well
    st.session_state.state.memory = ConversationBufferWindowMemory(
        memory_key="chat_history",
        k=10,
        return_messages=True
    )
    st.success("🔄 Chat and session reset!")


def render_sidebar():
    st.sidebar.title("🧠 AskYourDoc")
    st.sidebar.markdown("Upload your document and ask questions about it.")

    uploaded_file = st.sidebar.file_uploader("📄 Upload a document", type=["pdf", "docx", "txt"])

    if uploaded_file:
        current_file_name = uploaded_file.name
        prev_file_name = getattr(st.session_state.state.uploaded_file, 'name', None)

        if current_file_name != prev_file_name:
            st.session_state.state.vector_store = None
            process_uploaded_file(uploaded_file)
            st.session_state.state.uploaded_file = uploaded_file
            st.success("✅ Document uploaded and processed! You can now ask questions.")

    if st.sidebar.button("🧹 Reset Chat"):
        reset_chat()

def render_chat():
    st.header("🤖 PalmMind AI Assistant")
    for idx, msg in enumerate(st.session_state.state.chat_history):
        message(msg["content"], is_user=(msg["role"] == "user"), key=f"chat_{idx}_{msg['role']}")

def render_form_progress():
    if st.session_state.state.form.active:
        current_step = st.session_state.state.form.current_step + 1
        total_steps = len(AppointmentForm.FORM_STEPS)
        st.progress(current_step / total_steps)
        st.caption(f"Step {current_step} of {total_steps}")

def main():
    initialize_session()
    render_sidebar()
    render_chat()
    render_form_progress()

    user_input = st.text_input(
        "Type your message...",
        key="user_input",
        placeholder="Ask anything or say 'book appointment'"
    )

    def on_send_click():
        handle_user_input(st.session_state.user_input)
        st.session_state.user_input = ""

    st.button("Send", on_click=on_send_click)
    st.markdown("---")
    st.caption("PalmMind AI Assistant v1.0 | Chat with or without documents")

if __name__ == "__main__":
    main()
