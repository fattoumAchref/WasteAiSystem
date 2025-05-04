# ragwaste_service/response_service.py

from RagWaste import WikipediaAPIWrapper, GoogleBooksTool, is_waste_related, chain, task_planner
from utils.response_formatter import ResponseFormatter

async def handle_chat_response(prompt: str) -> str:
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

    return ResponseFormatter.format_full_response(
        wiki=wiki_content,
        books=books_content,
        rag=rag_content
    )
