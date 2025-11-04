from typing import TypedDict, Annotated, List, Optional
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import operator
from dotenv import load_dotenv
import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from together import Together

# ============ ENVIRONMENT SETUP ============
load_dotenv()

TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
FINE_TUNED_MODEL = os.getenv(
    "FINE_TUNED_MODEL", 
    "rahmanhabeeb360_24ae/Meta-Llama-3.1-8B-Instruct-Reference-ft-manslater-4c03ddd8"
)

os.environ["OPENAI_API_KEY"] = TOGETHER_API_KEY


# ============ LANGGRAPH SETUP ============

class ConversationState(TypedDict):
    """State that gets passed between nodes"""
    messages: Annotated[List, operator.add]


# Initialize LLM
llm = ChatOpenAI(
    base_url="https://api.together.xyz/v1",
    api_key=TOGETHER_API_KEY,
    model=FINE_TUNED_MODEL,
    temperature=0.8,
    max_tokens=100
)


def response_node(state: ConversationState) -> ConversationState:
    """Generate response using fine-tuned model"""
    messages = state.get("messages", [])
    
    system_prompt = """You are a brutally honest relationship advisor for guys. 
    You speak like a bro giving tough love. 
    to understand the situation before giving advice. Never sugarcoat. 
    Always give EXACT phrases to say.
    Always be caring towards women."""
    
    llm_messages = [SystemMessage(content=system_prompt)] + messages[-8:]
    response = llm.invoke(llm_messages)
    
    return {"messages": [AIMessage(content=response.content)]}


def create_advisor_graph():
    """Build and compile the conversation graph"""
    workflow = StateGraph(ConversationState)
    workflow.add_node("response", response_node)
    workflow.set_entry_point("response")
    workflow.add_edge("response", END)
    
    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)


class Manslater:
    def __init__(self):
        self.graph = create_advisor_graph()
        self.thread_id = "conversation_" + str(os.urandom(4).hex())
        self.config = {"configurable": {"thread_id": self.thread_id}}
    
    def send_message(self, user_message: str):
        """Send user message and get response"""
        input_data = {"messages": [HumanMessage(content=user_message)]}
        result = self.graph.invoke(input_data, self.config)
        
        # Return last AI message
        for msg in reversed(result["messages"]):
            if isinstance(msg, AIMessage):
                return msg.content
        
        return "Something went wrong, bro."


# ============ FASTAPI APP SETUP ============

app = FastAPI(title="Manslater API", version="1.0.0")

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Together client for direct API calls
together_client = Together(api_key=TOGETHER_API_KEY)

# Store active sessions
sessions = {}


# ============ REQUEST/RESPONSE MODELS ============

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    response: str
    session_id: str


class TranslateRequest(BaseModel):
    text: str


class TranslateResponse(BaseModel):
    translatedText: str


# ============ ENDPOINTS ============

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Welcome to Manslater API",
        "endpoints": {
            "chat": "/chat - Conversational endpoint with memory",
            "translate": "/translate - Single turn translation",
            "health": "/health - Health check",
            "docs": "/docs - API documentation"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "model": FINE_TUNED_MODEL}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Handle chat messages with conversation memory
    Uses LangGraph for stateful conversations
    """
    try:
        # Get or create session
        session_id = request.session_id or f"session_{os.urandom(8).hex()}"
        
        if session_id not in sessions:
            sessions[session_id] = Manslater()
        
        advisor = sessions[session_id]
        
        # Get response
        response_text = advisor.send_message(request.message)
        
        return ChatResponse(
            response=response_text,
            session_id=session_id
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat error: {str(e)}")


@app.post("/translate", response_model=TranslateResponse)
async def translate_text(request: TranslateRequest):
    """
    Single-turn translation endpoint (no conversation memory)
    Direct API call to Together AI
    """
    if not request.text:
        raise HTTPException(status_code=400, detail="No text provided")

    if not FINE_TUNED_MODEL:
        raise HTTPException(status_code=500, detail="Fine-tuned model not configured")

    try:
        response = together_client.chat.completions.create(
            model=FINE_TUNED_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a brutally honest relationship advisor for guys. "
                        "You speak like a bro giving tough love. "
                        "to understand the situation before giving advice. Never sugarcoat. "
                        "Always give EXACT phrases to say."
                    )
                },
                {"role": "user", "content": request.text}
            ],
            max_tokens=100,
            temperature=0.8
        )

        translated_output = response.choices[0].message.content.strip()
        return TranslateResponse(translatedText=translated_output)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Translation failed: {str(e)}")


@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    """Delete a conversation session"""
    if session_id in sessions:
        del sessions[session_id]
        return {"message": "Session deleted successfully"}
    raise HTTPException(status_code=404, detail="Session not found")


@app.get("/sessions")
async def list_sessions():
    """List all active session IDs"""
    return {
        "active_sessions": list(sessions.keys()),
        "total": len(sessions)
    }


# ============ RUN SERVER ============

if __name__ == "__main__":
    import uvicorn
    print("=" * 60)
    print("🚀 Starting Unified Manslater API Server")
    print("=" * 60)
    print("Server running at: http://localhost:8000")
    print("API Documentation: http://localhost:8000/docs")
    print("Interactive API: http://localhost:8000/redoc")
    print("=" * 60)
    print("\nAvailable Endpoints:")
    print("  POST /chat        - Conversational chat with memory")
    print("  POST /translate   - Single-turn translation")
    print("  GET  /health      - Health check")
    print("  DELETE /session/{id} - Delete session")
    print("  GET  /sessions    - List active sessions")
    print("=" * 60)
    
    uvicorn.run(app, host="0.0.0.0", port=8000)