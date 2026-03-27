from fastapi import APIRouter
from pydantic import BaseModel
from app.services.agent_service import AgentService
from enum import Enum

class ModelEnum(str, Enum):
    LLAMA_70B = "llama-3.3-70b-versatile"
    LLAMA_8B = "llama-3.1-8b-instant"
    QWEN = "qwen/qwen3-32b"
    GPT_OSS = "openai/gpt-oss-120b"
    LLAMA_SCOUT = "meta-llama/llama-4-scout-17b-16e-instruct"
    LLAMA_VISION = "llama-3.2-11b-vision-preview"

class UsecaseEnum(str, Enum):
    BASIC = "Basic Chatbot"
    ADVANCED = "Advanced Chatbot"
    MULTI_AGENT = "Multi-Agent Chatbot"
    DOCUMENT_QA = "Document QA"
router = APIRouter()
service = AgentService()

class ChatRequest(BaseModel):
    user_id: str
    conversation_id: str
    message: str
    usecase: UsecaseEnum
    model: ModelEnum

@router.post("/chat")
def chat(req: ChatRequest):

    response = service.run(
        user_id=req.user_id,
        conversation_id=req.conversation_id,
        message=req.message,
        usecase=req.usecase.value,
        model_name=req.model.value
    )

    return {"response": response}