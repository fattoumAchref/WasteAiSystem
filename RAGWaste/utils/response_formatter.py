class ResponseFormatter:
    @staticmethod
    def format_wikipedia(response: str) -> str:
        """Clean Wikipedia responses"""
        if not response:
            return ""
            
        if "Summary:" in response:
            summary = response.split("Summary:")[1].split("Page:")[0].strip()
            return f"📚:\n{summary[:800]}{'...' if len(summary) > 800 else ''}"
        return f"📚 Wikipedia Info:\n{response[:1000]}{'...' if len(response) > 1000 else ''}"

    @staticmethod
    def format_books(response: str) -> str:
        """Format book results"""
        if not response or "No relevant books" in response:
            return ""
        return "📖 Recommended Books:\n" + response

    @staticmethod
    def format_rag(response: str) -> str:
        """Clean RAG responses"""
        if not response:
            return ""
            
        # Remove any garbled text after the actual response
        clean_response = response.split("Based on this document")[0]
        clean_response = clean_response.split("Q:")[0]
        clean_response = clean_response.split("Paragraphs.")[0]
        
        return "♻️ Recycling Process:\n" + clean_response.strip()

    @classmethod
    def format_full_response(cls, wiki: str, books: str, rag: str) -> str:
        """Combine all formatted sections"""
        parts = []
        if wiki:
            parts.append(cls.format_wikipedia(wiki))
        if books:
            parts.append(cls.format_books(books))
        if rag:
            parts.append(cls.format_rag(rag))
            
        return "\n\n".join(parts) if parts else "No information found"