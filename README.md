# RAG_CHATBOT_FOR_DOCUMENTS

An intelligent assistant powered by AutoGen + RAG + FAISS + Streamlit to answer your questions from PDF, Word, or PowerPoint files using Azure OpenAI embeddings.

✨ Features
📂 Upload your own documents (PDF, DOCX, PPTX)

🧠 Chat with your documents using natural language

🔍 Context-aware answers powered by Retrieval-Augmented Generation

⚡ AutoGen agents handle multi-step reasoning

🔒 Powered by your Azure OpenAI API for embeddings and OpenAI LLM for reasoning

🖥️ Beautiful and intuitive UI via Streamlit

├── app.py               # Streamlit frontend & AutoGen integration
├── main_chat.py         # Headless AutoGen test runner
├── rag_index.py         # RAG logic: text extraction, embedding, FAISS indexing
├── tools.py             # retrieve_doc_context tool for document querying (not shown)
├── .env                 # Store API keys and config (not included)
└── rag_faiss_store/     # Folder for storing FAISS index
