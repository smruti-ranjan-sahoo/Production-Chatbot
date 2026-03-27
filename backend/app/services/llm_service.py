import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

class LLMService:

    def get_llm(self, model_name: str):
        return ChatGroq(
            api_key=os.getenv("GROQ_API_KEY"),
            model=model_name,
            temperature=0.2
        )