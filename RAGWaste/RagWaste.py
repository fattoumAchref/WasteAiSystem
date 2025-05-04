
import os
import streamlit as st
from glob import glob
import json
from datetime import datetime
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.chat_models import ChatOllama
from langchain.agents import Tool, initialize_agent
from langchain.tools import BaseTool
from langchain.prompts import PromptTemplate
from agents.retrieval_agent import RetrievalAgent
from agents.task_planner import TaskPlanner
from langchain_community.utilities import WikipediaAPIWrapper
from tools.google_books import GoogleBooksTool
from utils.response_formatter import ResponseFormatter
import time
# ================================
# 🔹 HELPER FUNCTION FOR ANIMATION
# ================================

def typewriter_effect(text: str, speed: float = 0.01):
    """Display text with a typewriter effect."""
    container = st.empty()
    current_text = ""
    for char in text:
        current_text += char
        container.markdown(current_text)
        time.sleep(speed)
    return container
# ================================
# 🔹 1. INITIALIZATIONS
# ================================

# Folder paths
folder_path = 'C:/Users/fatto/Desktop/RAGWaste/data'
embeddings_file = 'C:/Users/fatto/Desktop/RAGWaste/faiss_index'
processed_files_file = 'C:/Users/fatto/Desktop/RAGWaste/processed1_files.json'

# Initialize embeddings model
embeddings = HuggingFaceEmbeddings(
    model_name='all-mpnet-base-v2',
    model_kwargs={"device": "cuda"},
    encode_kwargs={
        "normalize_embeddings": False,  # Faster processing
        "batch_size": 32  # Optimized for CPU
    }
    )

# ================================
# 🔹 2. PDF PROCESSING & VECTORSTORE
# ================================

@st.cache_resource
def initialize_vectorstore():
    processed_files = set()
    if os.path.exists(processed_files_file):
        with open(processed_files_file, 'r') as f:
            processed_files = set(json.load(f))

    pdf_files = glob(f"{folder_path}/*.pdf")
    new_files = [f for f in pdf_files if f not in processed_files]

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    vectorstore = None
    if os.path.exists(embeddings_file) and os.listdir(embeddings_file):
        try:
            vectorstore = FAISS.load_local(embeddings_file, embeddings, allow_dangerous_deserialization=True)
            st.sidebar.success("Loaded existing vector store")
        except Exception as e:
            st.sidebar.error(f"Error loading vector store: {e}")
    else:
        st.sidebar.info("Creating new vector store")

    if new_files:
        for file_path in new_files:
            try:
                loader = PyPDFLoader(file_path)
                pages = loader.load_and_split(text_splitter)
                if vectorstore:
                    vectorstore.add_documents(pages)
                else:
                    vectorstore = FAISS.from_documents(pages, embeddings)
                processed_files.add(file_path)
            except Exception as e:
                st.sidebar.error(f"Error processing {file_path}: {e}")

        if vectorstore:
            vectorstore.save_local(embeddings_file)
            with open(processed_files_file, 'w') as f:
                json.dump(list(processed_files), f)
            st.sidebar.success(f"Processed {len(new_files)} new files")

    return vectorstore or FAISS.from_texts(["Placeholder"], embeddings)

vectorstore = initialize_vectorstore()

# ================================
# 🔹 3. MEMORY AND TOOLS SETUP
# ================================

llm = ChatOllama(
    model="phi3:mini",
    
)

task_planner = TaskPlanner(llm=llm)
# Initialize memories with proper configuration
agent_memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    input_key="input",
    output_key="output"
)

chain_memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    output_key="answer"
)

# Tools setup
tools = [
    Tool(
        name="Wikipedia",
        func=WikipediaAPIWrapper().run,
        description="Useful for general knowledge about waste management concepts, materials, and processes"
    ),
    GoogleBooksTool(
        name="Google_Books",
        description="Useful for finding authoritative books and publications about waste management"
    )
]

# Initialize agent
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent="structured-chat-zero-shot-react-description",
    memory=agent_memory,
    verbose=True,
    handle_parsing_errors=True  # Important for reliability
)

# ================================
# 🔹 4. CHAIN SETUP WITH PROMPTS
# ================================

pre_prompt = """I am a WASTE MANAGEMENT SPECIALIST. My knowledge is STRICTLY LIMITED to:
- Recycling processes ♻️
- Waste disposal methods 🗑️  
- Composting techniques 🍂
- Waste policies 📜

"""

condense_prompt = PromptTemplate(
    input_variables=["chat_history", "question"],
    template=f"""{pre_prompt}
    
Rephrase this question to focus specifically on waste management aspects.
If completely unrelated, return 'OFFTOPIC'.

Chat History: {{chat_history}}
Question: {{question}}
Rephrased Waste-Focused Question:"""
)

chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=RetrievalAgent(vectorstore=vectorstore),
    memory=chain_memory,
    condense_question_prompt=condense_prompt,
    return_source_documents=True,
    verbose=True
)

# ================================
# 🔹 5. STREAMLIT UI
# ================================

st.sidebar.title('Waste Solutions AI - Waste AI Assistant')
def verify_relevance(documents: list[Document], query: str) -> list[Document]:
    """Ensure documents mention the queried material in first 3 sentences."""
    query_material = query.split()[0].lower()
    relevant_docs = []
    
    for doc in documents:
        first_lines = ' '.join(doc.page_content.split('.')[:3]).lower()
        if query_material in first_lines:
            relevant_docs.append(doc)
        elif len(relevant_docs) < 2:  # Keep max 2 backup docs
            relevant_docs.append(doc)
            
    return relevant_docs



def verify_response_material(query: str, response: str) -> bool:
    """Check if response actually mentions the queried material"""
    material = TaskPlanner._extract_material(query)
    return material.lower() in response.lower()

def is_waste_related(query: str) -> bool:
    waste_keywords = {
        'recycle', 'waste', 'dispose', 'compost', 'landfill',
        'incinerat', 'hazardous', 'e-waste', 'plastic', 'metal',
        'organic waste', 'policy', 'regulation', 'collection', 'oil',
        'glass', 'cardboard boxes', 'papers', 'medical', 'aluminium',
        'food'
    }
    return any(keyword in query.lower() for keyword in waste_keywords)

# Initialize session state for chat history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Main chat interface
prompt = st.text_input('Enter your question:', key="user_input")

if st.button("Send") and prompt:
    user_message = {"role": "user", "content": prompt}
    st.session_state.chat_history.append(user_message)
    st.chat_message('user').markdown(prompt)
    
    if not is_waste_related(prompt):
        response = {"role": "assistant", "content": pre_prompt, "sources": []}
    else:
        with st.spinner("Researching your query..."):
            try:
                # Get the enhanced plan with tool recommendations
                plan = task_planner.plan(prompt)
                material = task_planner._extract_material(prompt)
                
                # Initialize response components
                # Process based on recommended tools
                # Initialize response components
                response_content = ""
                sources = []
                tools_used = []  # Initialize the list here

                # Process based on recommended tools
                if "wikipedia" in plan["tools"]:
                    wiki_result = tools[0].run(f"{material} waste recycling")
                    tools_used.append("wikipedia")
                    
                if "google_books" in plan["tools"]:
                    books_result = tools[1].run(f"{material} waste management")
                    tools_used.append("google_books")

                # Always include RAG results (retrieval tool)
                rag_response = chain({"question": task_planner._rewrite_query(prompt)})
                rag_result = rag_response['answer']
                tools_used.append("retrieval")
                sources = [doc.metadata["source"] for doc in rag_response["source_documents"]]

                # Format final response using ResponseFormatter
                response_content = ResponseFormatter.format_full_response(
                    wiki=wiki_result if "wikipedia" in tools_used else "",
                    books=books_result if "google_books" in tools_used else "",
                    rag=rag_result
                )

                response = {
                    "role": "assistant",
                    "content": response_content,
                    "sources": sources,
                    "tools_used": tools_used
                }
                
            except Exception as e:
                # Fallback to simple RAG if tools fail
                try:
                    rag_response = chain({"question": prompt})
                    response = {
                        "role": "assistant",
                        "content": ResponseFormatter.format_rag(rag_response["answer"]),
                        "sources": [doc.metadata["source"] for doc in rag_response["source_documents"]],
                        "tools_used": ["retrieval"]  # Explicitly defined here
                    }
                except:
                    response = {
                        "role": "assistant",
                        "content": f"Please try rephrasing your question about {material}",
                        "sources": [],
                        "tools_used": []  # Empty list for cases where no tools were used
                    }
    
    # Display response with tool attribution
    with st.chat_message("assistant"):
        response_container = typewriter_effect(response["content"])
    
    if response["tools_used"]:
        with st.expander(f"Methods used ({', '.join(response['tools_used'])})"):
            if response["sources"]:
                st.write("Sources:")
                st.write("\n".join([f"• {src}" for src in response["sources"]]))
            st.write("Powered by: " + ", ".join(
                [task_planner.get_tool_description(tool) for tool in response["tools_used"]]
            ))
    
    # Update memory (simplified)
#    memory_input = {
#        "input": prompt,
#        "output": response["content"],
#        "tools": response["tools_used"]
#    }
#    agent_memory.save_context(memory_input, {"output": response["content"]})
    agent_memory.save_context(
    {"input": prompt},  # Single key dictionary
    {"output": response["content"]}
    )
    chain_memory.save_context({"question": prompt}, {"answer": response["content"]})
    st.session_state.chat_history.append(response)

# Display sources if available
if st.session_state.chat_history and st.session_state.chat_history[-1]["sources"]:
    with st.expander("Sources"):
        for source in st.session_state.chat_history[-1]["sources"]:
            st.write(f"📄 {source}")

# ================================
# 🔹 6. CHAT EXPORT FUNCTIONALITY
# ================================

def export_chat_history():
    try:
        # Create export directory
        export_dir = os.path.join('C:/Users/fatto/Desktop/RAGWaste', 'chat_exports')
        os.makedirs(export_dir, exist_ok=True)
        
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        export_file = os.path.join(export_dir, f"chat_export_{timestamp}.txt")
        
        # Format chat history
        formatted_history = []
        for msg in st.session_state.chat_history:
            prefix = "👤 User" if msg["role"] == "user" else "🤖 Assistant"
            formatted_history.append(f"{prefix}: {msg['content']}")
            
            if msg["role"] == "assistant" and msg["sources"]:
                formatted_history.append("📚 Sources:")
                formatted_history.extend([f"  - {src}" for src in msg["sources"]])
        
        # Write to file
        with open(export_file, 'w', encoding='utf-8') as f:
            f.write("\n\n".join(formatted_history))
        
        return export_file
    
    except Exception as e:
        st.sidebar.error(f"Export failed: {str(e)}")
        return None

if st.sidebar.button("Export Chat"):
    exported_file = export_chat_history()
    if exported_file:
        st.sidebar.success(f"Chat exported to:\n{exported_file}")
        with open(exported_file, 'r', encoding='utf-8') as f:
            st.sidebar.text(f.read())

# Display conversation history
st.sidebar.subheader("Conversation History")
for msg in st.session_state.chat_history[-10:]:  # Show last 10 messages
    prefix = "👤" if msg["role"] == "user" else "🤖"
    st.sidebar.text(f"{prefix} {msg['content'][:50]}...")