from fastapi import FastAPI
import uvicorn
import consul
import socket
from pydantic import BaseModel
from typing import List, Dict
from agents.task_planner import TaskPlanner  # Make sure this path is correct
from services.response_service import handle_chat_response
# Mock LLM (Replace with actual chain or LLM later)
class MockLLM:
    def __call__(self, messages):
        return type("Response", (), {
            "content": '{"tasks": ["Collect used material", "Sort it properly", "Send to recycling facility"], "tools": ["retrieval", "wikipedia"]}'
        })

# Setup
app = FastAPI()
service_name = "ragwaste-service"
service_id = f"{service_name}-{socket.gethostname()}"
service_port = 8004
planner = TaskPlanner(llm=MockLLM())  # Inject your LLM or chain here
c = consul.Consul(host="localhost", port=8500)

@app.on_event("startup")
def register_service():
    c.agent.service.register(
        name=service_name,
        service_id=service_id,
        address=socket.gethostbyname(socket.gethostname()),
        port=service_port,
        tags=["waste", "recycling", "rag"]
    )
    print(f"✅ Registered {service_name} to Consul")

@app.on_event("shutdown")
def deregister_service():
    c.agent.service.deregister(service_id)
    print(f"❌ Deregistered {service_name} from Consul")


# Request and response models
class ChatRequest(BaseModel):
    prompt: str

class ChatResponse(BaseModel):
    response: str

@app.post("/chat", response_model=ChatResponse)
async def chat_handler(request: ChatRequest):
    response = await handle_chat_response(request.prompt)
    return ChatResponse(response=response)

if __name__ == "__main__":
    uvicorn.run("ragwaste_service:app", host="0.0.0.0", port=service_port, reload=True)
