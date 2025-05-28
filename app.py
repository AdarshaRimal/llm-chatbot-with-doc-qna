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

st.set_page_config(page_title="AI Assistant", page_icon="🤖", layout="centered")


class SessionState:
    def __init__(self):
        self.chat_history = []
        self.uploaded_file = None
        self.vector_store = None
        self.qa_chain = None
        self.agent = None
        self.form = AppointmentForm()
        self.processing_lock = False
        self.document_processed = False  # Track document state
        self.current_document_name = None  # Track document name
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0.3,
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
        # Enhanced conversation memory
        self.memory = ConversationBufferWindowMemory(
            memory_key="chat_history",
            k=20,  # Increased memory capacity
            return_messages=True
        )
        self.in_form_mode = False


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

    system_msg = (
        "You are a helpful assistant with two capabilities:\n"
        "1. Answer general questions using your knowledge\n"
        "2. Book appointments when requested\n\n"
        "Rules:\n"
        "- If the user wants to book an appointment, use the BookAppointment tool\n"
        "- For all other questions, answer directly using your knowledge\n\n"
        "When using a tool, respond exactly as:\n"
        "Thought: <your thought>\n"
        "Action: <tool name>\n"
        "Action Input: <input>\n\n"
        "If you know the final answer, respond with:\n"
        "Final Answer: <your answer>\n\n"
        "Begin!\n"
    )

    return build_agent(
        llm=st.session_state.state.llm,
        tools=tools,
        system_message=system_msg,
        memory=st.session_state.state.memory
    )

def build_document_agent(retriever):
    doc_qa_chain = build_qa_chain(retriever)

    def document_qa_func(query: str) -> str:
        result = get_answer(doc_qa_chain, query)
        return result["answer"]

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

    system_msg = (
        f"You are analyzing: {st.session_state.state.current_document_name}\n\n"
        "You have access to two tools:\n"
        "1. DocumentQA - Use for ANY questions about the current document\n"
        "2. BookAppointment - Use to schedule appointments\n\n"
        "Rules:\n"
        "- For any question that might be answered by the document, use DocumentQA first\n"
        "- If DocumentQA returns 'I couldn't find relevant information', then answer using your own knowledge\n"
        "- If the user wants to book an appointment, use BookAppointment\n\n"
        "When using a tool, respond exactly as:\n"
        "Thought: <your thought>\n"
        "Action: <tool name>\n"
        "Action Input: <input>\n\n"
        "If you know the final answer, respond with:\n"
        "Final Answer: <your answer>\n\n"
        "Begin!\n"
    )

    return build_agent(
        llm=st.session_state.state.llm,
        tools=tools,
        system_message=system_msg,
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

        # Update document state
        st.session_state.state.vector_store = vector_store
        st.session_state.state.document_processed = True
        st.session_state.state.current_document_name = uploaded_file.name
        st.session_state.state.agent = build_document_agent(vector_store.as_retriever())

        # Add document context to memory
        st.session_state.state.memory.chat_memory.add_ai_message(
            f"📄 Ready to answer questions about: {uploaded_file.name}"
        )

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
        # Handle form interactions first
        if state.form.active:
            response, completed = state.form.handle_input(user_input)
            state.chat_history.append({"role": "assistant", "content": response})
            if completed:
                state.in_form_mode = False
            return

        # Check if we should start form
        if state.form.should_start(user_input):
            response = state.form.agent_trigger(user_input)
            state.chat_history.append({"role": "assistant", "content": response})
            state.in_form_mode = True
            return

        # Process normal queries
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
    state = st.session_state.state
    state.chat_history = []
    state.form = AppointmentForm()
    state.vector_store = None
    state.document_processed = False
    state.uploaded_file = None

    # Preserve document context if available
    if state.current_document_name:
        doc_name = state.current_document_name
        state.memory = ConversationBufferWindowMemory(
            memory_key="chat_history",
            k=20,
            return_messages=True
        )
        # Add document context back to memory
        state.memory.chat_memory.add_ai_message(
            f"Document context preserved: {doc_name}"
        )
    else:
        state.memory = ConversationBufferWindowMemory(
            memory_key="chat_history",
            k=20,
            return_messages=True
        )

    state.agent = build_base_agent()
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

    if st.sidebar.button("🧹 Reset Chat"):
        reset_chat()


def render_chat():
    st.header("🤖 AI Assistant")
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
    st.caption("AI Assistant  | Chat with or without documents")


if __name__ == "__main__":
    main()