from typing import TypedDict, Annotated, List, Optional
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import operator
from dotenv import load_dotenv
import os
import json
from fastapi import FastAPI, HTTPException, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from together import Together
import redis
from datetime import timedelta
import hashlib

# ============ ENVIRONMENT SETUP ============
load_dotenv()

TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
FINE_TUNED_MODEL = os.getenv(
    "FINE_TUNED_MODEL", 
    "rahmanhabeeb360_24ae/Meta-Llama-3.1-8B-Instruct-Reference-ft-manslater-4c03ddd8"
)
STYLE_REFINER_MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"  # Second LLM for style refinement

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_DB = int(os.getenv("REDIS_DB", 0))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", None)

# Rate limiting configuration
CHAT_LIMIT_PER_DEVICE = int(os.getenv("CHAT_LIMIT_PER_DEVICE", 5))
RATE_LIMIT_TTL = int(os.getenv("RATE_LIMIT_TTL", 86400))  # 24 hours default

os.environ["OPENAI_API_KEY"] = TOGETHER_API_KEY

# ============ REDIS SETUP ============

class RedisManager:
    """Manages Redis connections and operations"""
    
    def __init__(self):
        self.client = redis.Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            db=REDIS_DB,
            password=REDIS_PASSWORD,
            decode_responses=True,
            socket_connect_timeout=5,
            socket_timeout=5
        )
        self._test_connection()
    
    def _test_connection(self):
        """Test Redis connection on startup"""
        try:
            self.client.ping()
            print("✅ Redis connection established")
        except redis.ConnectionError as e:
            print(f"❌ Redis connection failed: {e}")
            raise
    
    # ============ RATE LIMITING METHODS ============
    
    def get_device_usage(self, device_id: str) -> int:
        """Get current usage count for a device"""
        key = f"rate_limit:device:{device_id}:chat_count"
        count = self.client.get(key)
        return int(count) if count else 0
    
    def increment_device_usage(self, device_id: str, ttl: int = RATE_LIMIT_TTL) -> int:
        """Increment usage count for a device and return new count"""
        key = f"rate_limit:device:{device_id}:chat_count"
        
        # Use pipeline for atomic operation
        pipe = self.client.pipeline()
        pipe.incr(key)
        pipe.expire(key, ttl)
        results = pipe.execute()
        
        return results[0]  # Return the new count
    
    def check_rate_limit(self, device_id: str, limit: int = CHAT_LIMIT_PER_DEVICE) -> dict:
        """Check if device has exceeded rate limit"""
        current_usage = self.get_device_usage(device_id)
        remaining = max(0, limit - current_usage)
        
        return {
            "allowed": current_usage < limit,
            "current_usage": current_usage,
            "limit": limit,
            "remaining": remaining
        }
    
    def reset_device_usage(self, device_id: str) -> bool:
        """Reset usage count for a device (admin function)"""
        key = f"rate_limit:device:{device_id}:chat_count"
        return self.client.delete(key) > 0
    
    def get_device_ttl(self, device_id: str) -> int:
        """Get TTL (time to live) for device rate limit"""
        key = f"rate_limit:device:{device_id}:chat_count"
        ttl = self.client.ttl(key)
        return ttl if ttl > 0 else 0
    
    # ============ CHAT SESSION METHODS ============
    
    def save_chat_message(self, session_id: str, role: str, content: str, ttl: int = 86400):
        """Save a chat message to Redis (24 hour default TTL)"""
        key = f"chat:{session_id}:messages"
        message = {
            "role": role,
            "content": content,
            "timestamp": str(os.urandom(4).hex())  # Simple timestamp replacement
        }
        self.client.rpush(key, json.dumps(message))
        self.client.expire(key, ttl)
    
    def get_chat_history(self, session_id: str, limit: int = 20) -> List[dict]:
        """Retrieve chat history from Redis"""
        key = f"chat:{session_id}:messages"
        messages = self.client.lrange(key, -limit, -1)
        return [json.loads(msg) for msg in messages]
    
    def delete_session(self, session_id: str) -> bool:
        """Delete a chat session from Redis"""
        key = f"chat:{session_id}:messages"
        return self.client.delete(key) > 0
    
    def get_all_sessions(self) -> List[str]:
        """Get all active session IDs"""
        pattern = "chat:*:messages"
        keys = self.client.keys(pattern)
        return [key.split(":")[1] for key in keys]
    
    def session_exists(self, session_id: str) -> bool:
        """Check if a session exists"""
        key = f"chat:{session_id}:messages"
        return self.client.exists(key) > 0
    
    # ============ TRANSLATION CACHE METHODS ============
    
    def get_translation_cache(self, text: str) -> Optional[str]:
        """Get cached translation result"""
        key = f"translation:{hash(text)}"
        cached = self.client.get(key)
        return cached if cached else None
    
    def set_translation_cache(self, text: str, translation: str, ttl: int = 3600):
        """Cache translation result (1 hour default TTL)"""
        key = f"translation:{hash(text)}"
        self.client.setex(key, ttl, translation)
    
    # ============ ANALYTICS METHODS ============
    
    def increment_request_count(self, endpoint: str):
        """Track API usage"""
        key = f"analytics:requests:{endpoint}"
        self.client.incr(key)
    
    def get_analytics(self) -> dict:
        """Get API usage statistics"""
        chat_count = self.client.get("analytics:requests:chat") or 0
        translate_count = self.client.get("analytics:requests:translate") or 0
        active_sessions = len(self.get_all_sessions())
        
        # Get rate limit statistics
        rate_limit_keys = self.client.keys("rate_limit:device:*:chat_count")
        total_devices = len(rate_limit_keys)
        
        return {
            "chat_requests": int(chat_count),
            "translate_requests": int(translate_count),
            "active_sessions": active_sessions,
            "tracked_devices": total_devices
        }


# Initialize Redis
redis_manager = RedisManager()


# ============ DEVICE IDENTIFICATION ============

def generate_device_id(request: Request, user_agent: Optional[str] = None) -> str:
    """
    Generate a unique device identifier from request headers
    Combines IP address, User-Agent, and other fingerprinting data
    """
    # Get IP address (handle proxy headers)
    ip = request.client.host if request.client else "unknown"
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        ip = forwarded.split(",")[0].strip()
    
    # Get user agent
    ua = user_agent or request.headers.get("user-agent", "unknown")
    
    # Get additional fingerprinting headers
    accept_lang = request.headers.get("accept-language", "")
    accept_encoding = request.headers.get("accept-encoding", "")
    
    # Create device fingerprint
    fingerprint = f"{ip}:{ua}:{accept_lang}:{accept_encoding}"
    
    # Hash it for privacy and consistency
    device_id = hashlib.sha256(fingerprint.encode()).hexdigest()[:16]
    
    return device_id


# ============ TWO-STEP LLM SETUP ============

# Initialize Together client for direct API calls
together_client = Together(api_key=TOGETHER_API_KEY)


def get_initial_response(user_message: str, conversation_history: List = None) -> str:
    """
    STEP 1: Get initial response from fine-tuned model
    """
    messages = []
    
    # Add conversation history if available
    if conversation_history:
        messages.extend(conversation_history[-8:])  # Last 8 messages for context
    
    # Add current user message
    messages.append({"role": "user", "content": user_message})
    
    try:
        response = together_client.chat.completions.create(
            model=FINE_TUNED_MODEL,
            messages=messages,
            max_tokens=150,
            temperature=0.7
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error in initial response: {e}")
        return "Error getting initial response"


def refine_response_style(initial_response: str, user_question: str) -> str:
    """
    STEP 2: Refine the response using Meta-Llama-3.1-8B-Instruct
    to match the desired style and tone
    """
    style_prompt = """You are a ruthless, sarcastic, emotionally-detached relationship drill-sergeant for men.

Your job: Take the advice given and rewrite it with the following style:

Tone rules:
• Brutal honesty, cold, mocking
• No empathy. No nurturing.
• Masculine, controlled, slightly toxic humor
• Snarky emojis allowed 😐🔥💀🙄
• No therapy language. No validation. No simping.

Format:
1. Start with a savage roast (1 line) about him even asking this
2. Then give the actual advice/line to say (short, sweet)
3. Keep it under 3-4 sentences total

Examples of the style:

Input: 'Just say - "Talk to me, please — I don't want to lose you." Be sweet, not needy. Don't get clingy, don't get ignored."
Output: "Bro, you really needs help with this? Embarrassing 💀. Just say - ' Talk to me, please — I don't want to lose you.' and call till she pick up."

Input : "Say - "I'd build you a worm palace and defend you from birds." Be absurd, be loyal, be extra."
Output: "She hit you with the worm question and you panicked? sad 🙄. Just Say - 'I'd build you a worm palace and defend you from birds.' Be absurd, be loyal, be extra"

Now rewrite this advice in that style:"""

    try:
        response = together_client.chat.completions.create(
            model=STYLE_REFINER_MODEL,
            messages=[
                {"role": "system", "content": style_prompt},
                {"role": "user", "content": f"User's question: {user_question}\n\nAdvice to rewrite: {initial_response}"}
            ],
            max_tokens=120,
            temperature=0.8
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error in style refinement: {e}")
        # Fallback to initial response if refinement fails
        return initial_response


# ============ LANGGRAPH SETUP (Modified) ============

class ConversationState(TypedDict):
    """State that gets passed between nodes"""
    messages: Annotated[List, operator.add]


# Initialize LLM (keeping for compatibility, but using Together client instead)
llm = ChatOpenAI(
    base_url="https://api.together.xyz/v1",
    api_key=TOGETHER_API_KEY,
    model=FINE_TUNED_MODEL,
    temperature=0.7,
    max_tokens=150
)


def response_node(state: ConversationState) -> ConversationState:
    """Generate response using TWO-STEP LLM process"""
    messages = state.get("messages", [])
    
    # Get the last user message
    user_message = ""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            user_message = msg.content
            break
    
    # Convert messages to dict format for Together API
    conversation_history = []
    for msg in messages[:-1]:  # All except last message
        if isinstance(msg, HumanMessage):
            conversation_history.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage):
            conversation_history.append({"role": "assistant", "content": msg.content})
    
    # STEP 1: Get initial response from fine-tuned model
    initial_response = get_initial_response(user_message, conversation_history)
    
    # STEP 2: Refine style with Meta-Llama-3.1-8B-Instruct
    refined_response = refine_response_style(initial_response, user_message)
    
    return {"messages": [AIMessage(content=refined_response)]}


def create_advisor_graph():
    """Build and compile the conversation graph"""
    workflow = StateGraph(ConversationState)
    workflow.add_node("response", response_node)
    workflow.set_entry_point("response")
    workflow.add_edge("response", END)
    
    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)


class Manslater:
    def __init__(self, session_id: str):
        self.graph = create_advisor_graph()
        self.session_id = session_id
        self.thread_id = f"thread_{session_id}"
        self.config = {"configurable": {"thread_id": self.thread_id}}
    
    def load_history_from_redis(self) -> List:
        """Load conversation history from Redis"""
        history = redis_manager.get_chat_history(self.session_id)
        messages = []
        for msg in history:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                messages.append(AIMessage(content=msg["content"]))
        return messages
    
    def send_message(self, user_message: str):
        """Send user message and get response"""
        # Save user message to Redis
        redis_manager.save_chat_message(self.session_id, "user", user_message)
        
        # Load history and create input
        history = self.load_history_from_redis()
        input_data = {"messages": history + [HumanMessage(content=user_message)]}
        
        # Get response from LLM (now uses two-step process)
        result = self.graph.invoke(input_data, self.config)
        
        # Extract AI response
        ai_response = None
        for msg in reversed(result["messages"]):
            if isinstance(msg, AIMessage):
                ai_response = msg.content
                break
        
        if not ai_response:
            ai_response = "Something went wrong, bro."
        
        # Save AI response to Redis
        redis_manager.save_chat_message(self.session_id, "assistant", ai_response)
        
        return ai_response


# ============ FASTAPI APP SETUP ============

app = FastAPI(title="Manslater API", version="2.1.0")

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
        "https://manslater.in",
        "https://www.manslater.in"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory cache for active Manslater instances
active_instances = {}


# ============ REQUEST/RESPONSE MODELS ============

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    response: str
    session_id: str
    rate_limit_info: dict


class TranslateRequest(BaseModel):
    text: str


class TranslateResponse(BaseModel):
    translatedText: str
    cached: bool = False


class RateLimitInfo(BaseModel):
    device_id: str
    current_usage: int
    limit: int
    remaining: int
    ttl_seconds: int


# ============ ENDPOINTS ============

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Welcome to Manslater API with Two-Step LLM Processing",
        "version": "2.1.0",
        "features": [
            "Two-step LLM: Fine-tuned model + Style refinement",
            "Redis-backed rate limiting and caching",
            f"{CHAT_LIMIT_PER_DEVICE} chat invocations per device per {RATE_LIMIT_TTL//3600} hours"
        ],
        "endpoints": {
            "chat": "/chat - Conversational endpoint (rate limited)",
            "translate": "/translate - Single turn translation",
            "health": "/health - Health check",
            "analytics": "/analytics - Usage statistics",
            "rate_limit": "/rate-limit - Check your rate limit status",
            "docs": "/docs - API documentation"
        }
    }


@app.get("/health")
@app.head("/health")
async def health_check():
    """Health check endpoint"""
    try:
        redis_manager.client.ping()
        redis_status = "connected"
    except:
        redis_status = "disconnected"
    
    return {
        "status": "healthy",
        "model": FINE_TUNED_MODEL,
        "style_refiner": STYLE_REFINER_MODEL,
        "redis": redis_status,
        "rate_limit": {
            "chat_limit": CHAT_LIMIT_PER_DEVICE,
            "ttl_hours": RATE_LIMIT_TTL // 3600
        }
    }


@app.get("/rate-limit", response_model=RateLimitInfo)
async def check_rate_limit(request: Request, user_agent: Optional[str] = Header(None)):
    """Check rate limit status for current device"""
    device_id = generate_device_id(request, user_agent)
    rate_info = redis_manager.check_rate_limit(device_id)
    ttl = redis_manager.get_device_ttl(device_id)
    
    return RateLimitInfo(
        device_id=device_id,
        current_usage=rate_info["current_usage"],
        limit=rate_info["limit"],
        remaining=rate_info["remaining"],
        ttl_seconds=ttl
    )


@app.post("/chat", response_model=ChatResponse)
async def chat(
    request_body: ChatRequest, 
    request: Request,
    user_agent: Optional[str] = Header(None)
):
    """
    Handle chat messages with conversation memory (stored in Redis)
    Uses TWO-STEP LLM: Fine-tuned model + Style refinement
    Rate limited to 5 invocations per device
    """
    try:
        # Generate device ID
        device_id = generate_device_id(request, user_agent)
        
        # Check rate limit BEFORE processing
        rate_info = redis_manager.check_rate_limit(device_id)
        
        if not rate_info["allowed"]:
            ttl = redis_manager.get_device_ttl(device_id)
            hours_remaining = ttl // 3600
            minutes_remaining = (ttl % 3600) // 60
            
            rate_limit_message = f"You have used all {CHAT_LIMIT_PER_DEVICE} chat invocations. Please try again in {hours_remaining}h {minutes_remaining}m."
            
            # Get or create session to save the rate limit message
            session_id = request_body.session_id or f"session_{os.urandom(8).hex()}"
            
            # Save user message
            redis_manager.save_chat_message(session_id, "user", request_body.message)
            
            # Save rate limit message as assistant response
            redis_manager.save_chat_message(session_id, "assistant", rate_limit_message)
            
            raise HTTPException(
                status_code=429,
                detail={
                    "error": "Rate limit exceeded",
                    "message": rate_limit_message,
                    "current_usage": rate_info["current_usage"],
                    "limit": rate_info["limit"],
                    "reset_in_seconds": ttl,
                    "device_id": device_id
                }
            )
        
        # Track request
        redis_manager.increment_request_count("chat")
        
        # Get or create session
        session_id = request_body.session_id or f"session_{os.urandom(8).hex()}"
        
        # Get or create Manslater instance
        if session_id not in active_instances:
            active_instances[session_id] = Manslater(session_id)
        
        advisor = active_instances[session_id]
        
        # Get response (THIS COUNTS AS 1 INVOCATION - now uses two-step LLM)
        response_text = advisor.send_message(request_body.message)
        
        # Increment device usage AFTER successful invocation
        new_usage = redis_manager.increment_device_usage(device_id)
        
        # Get updated rate limit info
        updated_rate_info = redis_manager.check_rate_limit(device_id)
        ttl = redis_manager.get_device_ttl(device_id)
        
        return ChatResponse(
            response=response_text,
            session_id=session_id,
            rate_limit_info={
                "current_usage": updated_rate_info["current_usage"],
                "limit": updated_rate_info["limit"],
                "remaining": updated_rate_info["remaining"],
                "reset_in_seconds": ttl
            }
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat error: {str(e)}")


@app.post("/translate", response_model=TranslateResponse)
async def translate_text(request: TranslateRequest):
    """
    Single-turn translation endpoint with Redis caching
    Uses TWO-STEP LLM processing
    NO RATE LIMITING - Free to use
    """
    if not request.text:
        raise HTTPException(status_code=400, detail="No text provided")

    try:
        # Track request
        redis_manager.increment_request_count("translate")
        
        # Check cache first
        cached_translation = redis_manager.get_translation_cache(request.text)
        if cached_translation:
            return TranslateResponse(
                translatedText=cached_translation,
                cached=True
            )
        
        # STEP 1: Get initial response from fine-tuned model
        initial_response = get_initial_response(request.text)
        
        # STEP 2: Refine style
        refined_response = refine_response_style(initial_response, request.text)
        
        # Cache the result
        redis_manager.set_translation_cache(request.text, refined_response)
        
        return TranslateResponse(
            translatedText=refined_response,
            cached=False
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Translation failed: {str(e)}")


@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    """Delete a conversation session from Redis"""
    deleted = redis_manager.delete_session(session_id)
    
    # Also remove from active instances
    if session_id in active_instances:
        del active_instances[session_id]
    
    if deleted:
        return {"message": "Session deleted successfully"}
    raise HTTPException(status_code=404, detail="Session not found")


@app.get("/sessions")
async def list_sessions():
    """List all active session IDs from Redis"""
    sessions = redis_manager.get_all_sessions()
    return {
        "active_sessions": sessions,
        "total": len(sessions)
    }


@app.get("/analytics")
async def get_analytics():
    """Get API usage analytics from Redis"""
    return redis_manager.get_analytics()


@app.delete("/cache/translations")
async def clear_translation_cache():
    """Clear all translation cache"""
    try:
        pattern = "translation:*"
        keys = redis_manager.client.keys(pattern)
        if keys:
            redis_manager.client.delete(*keys)
        return {"message": f"Cleared {len(keys)} cached translations"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cache clear failed: {str(e)}")


# ============ ADMIN ENDPOINTS (for testing/management) ============

@app.post("/admin/reset-device-limit")
async def reset_device_limit(
    request: Request,
    user_agent: Optional[str] = Header(None),
    admin_key: str = Header(None)
):
    """
    Reset rate limit for current device (admin only)
    Requires admin_key in header
    """
    # Simple admin key check (use proper auth in production)
    ADMIN_KEY = os.getenv("ADMIN_KEY", "your-secret-admin-key")
    
    if admin_key != ADMIN_KEY:
        raise HTTPException(status_code=403, detail="Unauthorized")
    
    device_id = generate_device_id(request, user_agent)
    reset = redis_manager.reset_device_usage(device_id)
    
    if reset:
        return {"message": "Device rate limit reset successfully", "device_id": device_id}
    return {"message": "No rate limit found for device", "device_id": device_id}


# ============ RUN SERVER ============

if __name__ == "__main__":
    import uvicorn
    print("=" * 60)
    print("🚀 Starting Manslater API v2.1.0")
    print("   Two-Step LLM Processing Active")
    print("=" * 60)
    print("Server running at: http://localhost:8000")
    print("API Documentation: http://localhost:8000/docs")
    print("=" * 60)
    print("\nLLM Configuration:")
    print(f"  Step 1 (Content): {FINE_TUNED_MODEL}")
    print(f"  Step 2 (Style): {STYLE_REFINER_MODEL}")
    print("=" * 60)
    print("\nRedis Configuration:")
    print(f"  Host: {REDIS_HOST}")
    print(f"  Port: {REDIS_PORT}")
    print("=" * 60)
    print("\nRate Limiting:")
    print(f"  Chat Limit: {CHAT_LIMIT_PER_DEVICE} invocations per device")
    print(f"  Reset Period: {RATE_LIMIT_TTL // 3600} hours")
    print(f"  Translation: Unlimited (cached)")
    print("=" * 60)
    
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)