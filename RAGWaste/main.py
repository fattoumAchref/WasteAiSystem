# main.py

import uvicorn
import consul
import socket
import app  # 👉 Import the real FastAPI app from app.py

service_name = "ragwaste-service"
service_id = f"{service_name}-{socket.gethostname()}"
service_port = 8001

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

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=service_port, reload=True)
