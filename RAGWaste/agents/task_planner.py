from langchain.schema import HumanMessage, SystemMessage
import json
import re
from typing import Dict, List

class TaskPlanner:
    def __init__(self, llm):
        self.llm = llm
        self.common_materials = {
            'glass', 'plastic', 'metal', 'paper', 'cardboard',
            'oil', 'batteries', 'electronics', 'organic', 'hazardous',
            'medical', 'aluminium', 'food', 'textiles'
        }
        self.available_tools = {
            'wikipedia': "Search general concepts and overviews",
            'google_books': "Find authoritative books and publications",
            'retrieval': "Search local knowledge base",
            'epa': "Check government regulations",
            'calculator': "Calculate costs and quantities"
        }

    def _is_process_query(self, query: str) -> bool:
        """Check if query already asks for steps/process"""
        process_phrases = {
            'steps', 'process', 'how to', 'method',
            'give me all', 'explain', 'describe'
        }
        query_lower = query.lower()
        return (any(phrase in query_lower for phrase in process_phrases) 
                or 'recycl' in query_lower)

    def _rewrite_query(self, query: str) -> str:
        """Only rewrite simple material queries"""
        if self._is_process_query(query):
            return query  # Keep original if already a process question
            
        base_material = query.split()[0].lower()
        if base_material in self.common_materials:
            return f"Explain the step-by-step recycling process for {base_material}"
        return query  # Return original if not a simple material

    def plan(self, query: str) -> Dict[str, List[str]]:
        """Enhanced planning with tool selection"""
        final_query = self._rewrite_query(query)
        material = self._extract_material(query)
        
        # Default fallback values
        default_tasks = [
            f"Research collection methods for {material}",
            f"Find sorting and processing techniques for {material}",
            f"Identify recycling facilities for {material}"
        ]
        default_tools = ['retrieval', 'wikipedia']
        
        try:
            messages = [
                SystemMessage(content="""You are a waste management research planner. For each query:
                1. Break into 3 sub-tasks
                2. Recommend 2-3 tools from: wikipedia, google_books, retrieval, epa, calculator
                Respond ONLY with this JSON format:
                {"tasks": ["task1", ...], "tools": ["tool1", ...]}"""),
                HumanMessage(content=f"""Query: {final_query}
                Material: {material}""")
            ]
            
            response = self.llm(messages).content
            plan = json.loads(response)
            
            # Validate tools
            valid_tools = [
                tool for tool in plan.get("tools", []) 
                if tool in self.available_tools
            ]
            
            return {
                "tasks": plan.get("tasks", default_tasks)[:3],  # Limit to 3 tasks
                "tools": valid_tools[:3] or default_tools  # Limit to 3 tools
            }
            
        except Exception as e:
            return {
                "tasks": default_tasks,
                "tools": default_tools
            }

    def _extract_material(self, query: str) -> str:
        """Extract base material from any query"""
        query_lower = query.lower()
        for material in self.common_materials:
            if material in query_lower:
                return material
        return query.split()[0].lower()

    def get_tool_description(self, tool_name: str) -> str:
        """Get human-readable tool description"""
        return self.available_tools.get(tool_name, "General research tool")