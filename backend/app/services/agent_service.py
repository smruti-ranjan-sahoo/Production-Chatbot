from langchain_core.messages import HumanMessage
from app.services.graph_service import GraphBuilder
from app.services.llm_service import LLMService

class AgentService:

    def __init__(self):
        self.llm_service = LLMService()

    def run(self, message: str, usecase: str, model_name: str):

        llm = self.llm_service.get_llm(model_name)

        graph = GraphBuilder(llm=llm).setup_graph(usecase)

        result = graph.invoke({
            "messages": [HumanMessage(content=message)]
        })

        return result["messages"][-1].content