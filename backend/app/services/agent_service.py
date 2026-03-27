from langchain_core.messages import HumanMessage, AIMessage
from app.services.graph_service import GraphBuilder
from app.services.llm_service import LLMService
from app.db.database import ChatDatabase


class AgentService:

    def __init__(self):
        self.llm_service = LLMService()
        self.db = ChatDatabase()

    def run(self, user_id, conversation_id, message, usecase, model_name):

        llm = self.llm_service.get_llm(model_name)

        # 🔥 LOAD HISTORY
        history = self.db.get_messages(conversation_id)

        messages = []
        for role, content in history:
            if role == "user":
                messages.append(HumanMessage(content=content))
            else:
                messages.append(AIMessage(content=content))

        # ➕ Add new message
        messages.append(HumanMessage(content=message))

        graph = GraphBuilder(llm=llm).setup_graph(usecase)

        result = graph.invoke({"messages": messages})

        ai_response = result["messages"][-1].content

        # 💾 SAVE BOTH
        self.db.add_message(conversation_id, "user", message)
        self.db.add_message(conversation_id, "assistant", ai_response)

        return ai_response