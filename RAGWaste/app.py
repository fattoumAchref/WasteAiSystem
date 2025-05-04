# app.py

from fastapi import FastAPI
from pydantic import BaseModel
from utils.response_formatter import ResponseFormatter
from RagWaste import WikipediaAPIWrapper, GoogleBooksTool, is_waste_related, task_planner, chain, agent, llm

app = FastAPI()

# Request schema
class PromptRequest(BaseModel):
    prompt: str

# Response schema
class ChatResponse(BaseModel):
    response: str

# Helper function
async def get_response(prompt, llm, chain, agent, task_planner):
    if not is_waste_related(prompt):
        return "I'm sorry, I can only assist with waste management questions."

    plan = task_planner.plan(prompt)
    material = task_planner._extract_material(prompt)

    wiki_content = ""
    books_content = ""
    rag_content = ""

    if "wikipedia" in plan["tools"]:
        wiki_content = WikipediaAPIWrapper().run(f"{material} waste recycling")

    if "google_books" in plan["tools"]:
        books_content = GoogleBooksTool().run(f"{material} waste management")

    rag_response = chain({"question": task_planner._rewrite_query(prompt)})
    rag_content = rag_response['answer']

    # Format the response nicely
    final_response = ResponseFormatter.format_full_response(
        wiki=wiki_content,
        books=books_content,
        rag=rag_content
    )

    return final_response

# FastAPI endpoint
@app.post("/chat/", response_model=ChatResponse)
async def chat(request: PromptRequest):
    prompt = request.prompt
    response = await get_response(prompt, llm, chain, agent, task_planner)
    return ChatResponse(response=response)
