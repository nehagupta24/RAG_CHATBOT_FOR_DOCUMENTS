import os
from dotenv import load_dotenv
load_dotenv()

from langchain_community.vectorstores import FAISS
from langchain_openai import AzureOpenAIEmbeddings
from autogen import AssistantAgent, register_function
from langchain_openai import ChatOpenAI

# Load Azure OpenAI and OpenAI configurations from .env
AZURE_OPENAI_API_KEY = os.getenv('AZURE_OPENAI_API_KEY')
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
AZURE_OPENAI_EMBEDDING_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")

openai_api_key = os.getenv("OPENAI_API_KEY")
openai_model = os.getenv("OPENAI_MODEL")


doc_assistant = AssistantAgent(
    name="DocAssistant",
    llm_config={
        "config_list": [
            {
                "model": openai_model,
                "api_key": openai_api_key
            }
        ]
    }
)

def retrieve_doc_context(query: str) -> str:
    """
    Retrieve relevant context from FAISS vector store built from any document type (PDF, DOCX, PPTX).
    """
    embeddings = AzureOpenAIEmbeddings(
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_API_KEY,
        api_version=AZURE_OPENAI_API_VERSION,
        azure_deployment=AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
    )

    # Load FAISS index (supports any doc type that was processed into the store)
    db = FAISS.load_local("rag_faiss_store", embeddings, allow_dangerous_deserialization=True)
    docs = db.similarity_search(query, k=3)  # get top 3 relevant chunks

    print("Similar Documents Found:")
    for doc in docs:
        print(doc)

    return "\n\n".join([doc.page_content for doc in docs])


register_function(
    retrieve_doc_context,
    caller=doc_assistant,
    executor=doc_assistant,
    description="Retrieve relevant context from the indexed documents (PDF, DOCX, PPTX) based on the user's query."
)


if __name__ == "__main__":
    query = "What are the education of neha?"
    context = retrieve_doc_context(query)
    print("Retrieved Context:\n", context)
