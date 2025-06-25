import os
from dotenv import load_dotenv
from autogen import AssistantAgent, UserProxyAgent
from tools import retrieve_doc_context  # UPDATED function from your tools
from langchain_openai import ChatOpenAI 

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
openai_model = os.getenv("OPENAI_MODEL")

llm_config = {
    "config_list": [
        {
            "model": openai_model,
            "api_key": openai_api_key
        }
    ]
}

def is_termination_msg(msg):
    return msg.get("content") and "TERMINATE" in msg["content"]

# Updated AssistantAgent: for ANY document, not limited to legal
doc_assistant = AssistantAgent(
    name="DocAssistant",
    system_message=(
        "You are a helpful document assistant that retrieves relevant information from documents (PDF, DOCX, PPTX).\n"
        "Based on the retrieved context, summarize and answer the user's query accurately.\n"
        "After answering the query, always respond with 'TERMINATE' to end the chat."
    ),
    llm_config=llm_config
)

user = UserProxyAgent(
    name="User",
    llm_config=False,
    human_input_mode="NEVER",  # no manual user input, all automated
    is_termination_msg=is_termination_msg,
    code_execution_config={"use_docker": False}
)

# Register the function to both agents
doc_assistant.register_for_llm(
    name="retrieve_doc_context",
    description="Retrieve relevant context from indexed documents (PDF, DOCX, PPTX) based on the query."
)(retrieve_doc_context)

user.register_for_execution(
    name="retrieve_doc_context"
)(retrieve_doc_context)

if __name__ == "__main__":
    # Initiate the conversation
    user.initiate_chat(
        doc_assistant,
        message="IS Neha know Machine learning?"
    )
