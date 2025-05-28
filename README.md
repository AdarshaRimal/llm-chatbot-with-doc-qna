# 🤖 AI Assistant

It is an AI-powered document chatbot built using LangChain, Gemini, and Streamlit. It allows users to upload documents and ask questions, get summaries, and even book appointments via a simple chat interface.

🎥 **YouTube Demo**: [Watch here](https://youtu.be/zz36oq5EvEo)

---

## 🚀 Features

- Upload documents (PDF, TXT, DOCX)
- Ask questions from your document
- Get summaries with one prompt
- Book appointments via chat
- Persistent chat memory
- Fast and lightweight

---

## 🛠 Tech Stack

- **Frontend**: Streamlit
- **LLM**: Google Gemini (via LangChain)
- **Vector Store**: FAISS
- **Document Parser**: Unstructured, PyMuPDF, python-docx
- **Memory**: ConversationBufferMemory

---

## 📁 Folder Structure

```
chatbot/
├── app.py                  # Streamlit app               
├── .env                    # Gemini API key
├── requirements.txt
├── chatbot/
│   ├── document_loader.py
        agent.py
│   ├── vector_store.py
│   ├── qa_chain.py
│   ├── form_handler.py
```

---

## ⚙️ Setup Instructions

### 1. Clone the Repo

```bash
git clone https://github.com/AdarshaRimal/llm-chatbot-with-doc-qna.git
cd llm-chatbot-with-dic-qna
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Add API Key

Create a `.env` file in the root folder:

```
GOOGLE_API_KEY=your_gemini_api_key_here
```

Get your key from: https://makersuite.google.com/app

### 4. Run the App

```bash
streamlit run app.py
```

---

## 💡 How to Use

- Upload your document from the sidebar
- Ask:
  - `"Summarize the document"`
  - `"What are the key takeaways?"`
  - `"book appointment"` to trigger form
- Get accurate, human-like answers

---

## 📺 Demo

Watch the full demo here:  
👉 https://youtu.be/zz36oq5EvEo

---



## 🙌 Credits

Built with ❤️ using LangChain, Gemini, Streamlit, and FAISS.
