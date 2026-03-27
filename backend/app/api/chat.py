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
router = APIRouter()
service = AgentService()

class ChatRequest(BaseModel):
    message: str
    usecase: str
    model: ModelEnum   # 👈 NEW

@router.post("/chat")
def chat(req: ChatRequest):
    response = service.run(
        message=req.message,
        usecase=req.usecase,
        model_name=req.model
    )
    return {"response": response}