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
from fastapi import UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from together import Together
import redis
from datetime import timedelta
import hashlib
import random
import asyncio
import time
from collections import defaultdict
from functools import wraps

# ============ ENVIRONMENT SETUP ============
load_dotenv()

TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
FINE_TUNED_MODEL = os.getenv(
    "FINE_TUNED_MODEL", 
    "rahmanhabeeb360_24ae/Meta-Llama-3.1-8B-Instruct-Reference-ft-manslater-4c03ddd8"
)
ROAST_MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_DB = int(os.getenv("REDIS_DB", 0))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", None)

CHAT_LIMIT_PER_DEVICE = int(os.getenv("CHAT_LIMIT_PER_DEVICE", 5))
RATE_LIMIT_TTL = int(os.getenv("RATE_LIMIT_TTL", 86400))

# API Timeouts
TOGETHER_API_TIMEOUT = int(os.getenv("TOGETHER_API_TIMEOUT", 30))  # 30 seconds timeout
MAX_CONCURRENT_REQUESTS = int(os.getenv("MAX_CONCURRENT_REQUESTS", 50))  # Global rate limit

# Memory management
MAX_ACTIVE_INSTANCES = int(os.getenv("MAX_ACTIVE_INSTANCES", 1000))  # Limit active instances
INSTANCE_CLEANUP_INTERVAL = int(os.getenv("INSTANCE_CLEANUP_INTERVAL", 300))  # 5 minutes

os.environ["OPENAI_API_KEY"] = TOGETHER_API_KEY

# ============ REDIS SETUP ============

class RedisManager:
    """Manages Redis connections and operations with connection pooling"""
    
    def __init__(self):
        # Use connection pool for better performance
        self.pool = redis.ConnectionPool(
            host=REDIS_HOST,
            port=REDIS_PORT,
            db=REDIS_DB,
            password=REDIS_PASSWORD,
            decode_responses=True,
            socket_connect_timeout=5,
            socket_timeout=5,
            max_connections=50,  # Connection pool size
            retry_on_timeout=True
        )
        self.client = redis.Redis(connection_pool=self.pool)
        self._test_connection()
    
    def _test_connection(self):
        try:
            self.client.ping()
            print("✅ Redis connection established")
        except redis.ConnectionError as e:
            print(f"❌ Redis connection failed: {e}")
            # Don't raise - allow graceful degradation
            print("⚠️  Continuing without Redis (rate limiting disabled)")
    
    def get_device_usage(self, device_id: str) -> int:
        key = f"rate_limit:device:{device_id}:chat_count"
        count = self.client.get(key)
        return int(count) if count else 0
    
    def increment_device_usage(self, device_id: str, ttl: int = RATE_LIMIT_TTL) -> int:
        key = f"rate_limit:device:{device_id}:chat_count"
        pipe = self.client.pipeline()
        pipe.incr(key)
        pipe.expire(key, ttl)
        results = pipe.execute()
        return results[0]
    
    def check_rate_limit(self, device_id: str, limit: int = CHAT_LIMIT_PER_DEVICE) -> dict:
        current_usage = self.get_device_usage(device_id)
        remaining = max(0, limit - current_usage)
        return {
            "allowed": current_usage < limit,
            "current_usage": current_usage,
            "limit": limit,
            "remaining": remaining
        }
    
    def reset_device_usage(self, device_id: str) -> bool:
        key = f"rate_limit:device:{device_id}:chat_count"
        return self.client.delete(key) > 0
    
    def get_device_ttl(self, device_id: str) -> int:
        key = f"rate_limit:device:{device_id}:chat_count"
        ttl = self.client.ttl(key)
        return ttl if ttl > 0 else 0
    
    def save_chat_message(self, session_id: str, role: str, content: str, ttl: int = 86400):
        key = f"chat:{session_id}:messages"
        message = {
            "role": role,
            "content": content,
            "timestamp": str(os.urandom(4).hex())
        }
        self.client.rpush(key, json.dumps(message))
        self.client.expire(key, ttl)
    
    def get_chat_history(self, session_id: str, limit: int = 20) -> List[dict]:
        key = f"chat:{session_id}:messages"
        messages = self.client.lrange(key, -limit, -1)
        return [json.loads(msg) for msg in messages]
    
    def delete_session(self, session_id: str) -> bool:
        key = f"chat:{session_id}:messages"
        return self.client.delete(key) > 0
    
    def get_all_sessions(self) -> List[str]:
        pattern = "chat:*:messages"
        keys = self.client.keys(pattern)
        return [key.split(":")[1] for key in keys]
    
    def session_exists(self, session_id: str) -> bool:
        key = f"chat:{session_id}:messages"
        return self.client.exists(key) > 0
    
    def get_translation_cache(self, text: str) -> Optional[str]:
        key = f"translation:{hash(text)}"
        cached = self.client.get(key)
        return cached if cached else None
    
    def set_translation_cache(self, text: str, translation: str, ttl: int = 3600):
        key = f"translation:{hash(text)}"
        self.client.setex(key, ttl, translation)
    
    def increment_request_count(self, endpoint: str):
        key = f"analytics:requests:{endpoint}"
        self.client.incr(key)
    
    def get_analytics(self) -> dict:
        chat_count = self.client.get("analytics:requests:chat") or 0
        translate_count = self.client.get("analytics:requests:translate") or 0
        active_sessions = len(self.get_all_sessions())
        rate_limit_keys = self.client.keys("rate_limit:device:*:chat_count")
        total_devices = len(rate_limit_keys)
        
        return {
            "chat_requests": int(chat_count),
            "translate_requests": int(translate_count),
            "active_sessions": active_sessions,
            "tracked_devices": total_devices
        }

redis_manager = RedisManager()

# ============ DEVICE IDENTIFICATION ============

def generate_device_id(request: Request, user_agent: Optional[str] = None) -> str:
    ip = request.client.host if request.client else "unknown"
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        ip = forwarded.split(",")[0].strip()
    
    ua = user_agent or request.headers.get("user-agent", "unknown")
    accept_lang = request.headers.get("accept-language", "")
    accept_encoding = request.headers.get("accept-encoding", "")
    
    fingerprint = f"{ip}:{ua}:{accept_lang}:{accept_encoding}"
    device_id = hashlib.sha256(fingerprint.encode()).hexdigest()[:16]
    
    return device_id

# ============ INTENT CLASSIFICATION ============

together_client = Together(api_key=TOGETHER_API_KEY)

# ============ GLOBAL RATE LIMITING ============

# Simple in-memory rate limiter for global request limiting
_concurrent_requests = 0
_max_concurrent = MAX_CONCURRENT_REQUESTS
_request_lock = asyncio.Lock() if hasattr(asyncio, 'Lock') else None

async def check_global_rate_limit():
    """Check if we've exceeded global concurrent request limit"""
    global _concurrent_requests
    if _request_lock:
        async with _request_lock:
            if _concurrent_requests >= _max_concurrent:
                raise HTTPException(
                    status_code=503,
                    detail=f"Server overloaded. Please try again in a moment. ({_concurrent_requests}/{_max_concurrent} concurrent requests)"
                )
            _concurrent_requests += 1
    else:
        # Fallback for older Python versions
        if _concurrent_requests >= _max_concurrent:
            raise HTTPException(
                status_code=503,
                detail=f"Server overloaded. Please try again in a moment."
            )
        _concurrent_requests += 1

async def release_global_rate_limit():
    """Release a request from the global counter"""
    global _concurrent_requests
    if _request_lock:
        async with _request_lock:
            _concurrent_requests = max(0, _concurrent_requests - 1)
    else:
        _concurrent_requests = max(0, _concurrent_requests - 1)

def classify_intent(user_message: str) -> str:
    """
    Classify if user wants TRANSLATION or just CHATTING
    Returns: "translate" or "chat"
    """
    
    classification_prompt = """You are an intent classifier for ManSlater, a savage relationship advice bot.

Analyze the user's message and determine:
- **TRANSLATE**: User is asking to interpret what a woman said/texted
- **CHAT**: User is just talking to ManSlater (greetings, complaints, casual chat, follow-ups)

TRANSLATION indicators:
- Contains quotes like "she said...", 'she texted...', "her message was..."
- Asking what something means: "what does X mean?", "decode this", "translate this"
- Direct female speech: any message with female dialogue/text
- Questions about her words/actions

CHAT indicators:
- Greetings: "hey", "yo", "sup", "what's up"
- Complaints: "bro she blocked me", "she's ignoring me again", "I messed up"
- Casual conversation: "what now?", "help me out", "I'm screwed"
- Follow-up questions: "ok but what if...", "should I...", "what do I do"
- Emotional venting: "I'm so confused", "I don't get it", "why is she like this"

EXAMPLES:

Input: "She said 'I'm fine'"
Output: TRANSLATE

Input: "she texted me 'we need to talk'"
Output: TRANSLATE

Input: "What does it mean when she says 'whatever'?"
Output: TRANSLATE

Input: "Hey bro"
Output: CHAT

Input: "What now?"
Output: CHAT

Input: "She's ignoring me again"
Output: CHAT

Input: "Bro I messed up"
Output: CHAT

Input: "Should I text her?"
Output: CHAT

Input: "I'm so confused about women"
Output: CHAT

Now classify this message:
USER MESSAGE: "{user_message}"

Respond with ONLY one word: TRANSLATE or CHAT"""

    try:
        response = together_client.chat.completions.create(
            model=ROAST_MODEL,
            messages=[
                {"role": "system", "content": "You are a binary classifier. Respond with only: TRANSLATE or CHAT"},
                {"role": "user", "content": classification_prompt.format(user_message=user_message)}
            ],
            max_tokens=5,
            temperature=0.3
        )
        
        intent = response.choices[0].message.content.strip().upper()
        
        # Validate response
        if "TRANSLATE" in intent:
            return "translate"
        elif "CHAT" in intent:
            return "chat"
        else:
            # Default: if uncertain, check for quotes as fallback
            if any(q in user_message.lower() for q in ['"', "'", "she said", "she texted", "her message"]):
                return "translate"
            return "chat"
            
    except Exception as e:
        print(f"Intent classification error: {e}")
        # Safe fallback: check for translation signals
        if any(q in user_message.lower() for q in ['"', "'", "she said", "she texted", "what does", "decode", "translate"]):
            return "translate"
        return "chat"

# ============ TRANSLATION MODE (Original) ============

def generate_roast(user_question: str) -> str:
    """Generate short roast for TRANSLATION mode"""
    roast_prompt = """You are a savage relationship coach who roasts men asking what women's words mean.

CRITICAL RULES:
1. MAXIMUM 3-4 WORDS (not sentences, WORDS!)
2. Must mock them for not understanding women
3. MUST include ONE emoji: 💀🔥😐🙄
4. NO explanations - just the roast

EXAMPLES:
Input: "She said 'I'm fine'"
Output: "Are you dumb? 💀"

Input: "She's ignoring me"
Output: "Weak move, bro 🙄"

Input: "What does 'whatever' mean?"
Output: "You serious? 😐"

Generate ONLY the roast (3-4 words + emoji):
{user_question}"""

    try:
        response = together_client.chat.completions.create(
            model=ROAST_MODEL,
            messages=[
                {"role": "system", "content": "Ultra-short savage roasts. Max 4 words + emoji."},
                {"role": "user", "content": roast_prompt.format(user_question=user_question)}
            ],
            max_tokens=15,
            temperature=0.9,
            top_p=0.95
        )
        
        roast = response.choices[0].message.content.strip()
        words = roast.split()
        if len(words) > 5:
            roast = ' '.join(words[:4])
        
        if not any(emoji in roast for emoji in ['💀', '🔥', '😐', '🙄']):
            roast += " 💀"
            
        return roast
        
    except Exception as e:
        print(f"Error generating roast: {e}")
        return "Seriously asking this? 💀"

def generate_advice(user_question: str, conversation_history: List = None) -> str:
    """Generate advice using fine-tuned model"""
    messages = []
    
    if conversation_history:
        messages.extend(conversation_history[-8:])
    
    messages.append({"role": "user", "content": user_question})
    
    try:
        response = together_client.chat.completions.create(
            model=FINE_TUNED_MODEL,
            messages=messages,
            max_tokens=150,
            temperature=0.7
        )
        advice = response.choices[0].message.content.strip()
        
        if random.random() < 0.8:
            advice = advice.replace("\n", " ")
        
        return advice
        
    except Exception as e:
        print(f"Error generating advice: {e}")
        return "just say - \"Talk to me when you're ready.\""

# ============ CHAT MODE (New) ============

def generate_savage_chat_response(user_message: str, conversation_history: List = None) -> str:
    """
    Generate savage ManSlater responses for casual chat
    Mocks the user for coming back, complaining, being clueless about women
    """
    
    chat_prompt = """You are ManSlater - a brutally savage relationship coach who roasts men who don't understand women.

The user is NOT asking to translate what a woman said. They're just CHATTING with you - complaining, venting, asking for general help, or greeting you.

YOUR PERSONALITY:
- Mock them for being back AGAIN with woman problems
- Shame them for not knowing how women work
- Short, punchy responses (1-2 sentences MAX)
- Always include ONE emoji: 💀🔥😐🙄😤
- Act like you're annoyed they need your help but you'll roast them anyway

RESPONSE STYLE:
- If they greet you: "Back again crying? 💀 What happened now?"
- If they complain: "And you're surprised? 😐 You did this to yourself."
- If they ask for help: "You really need me to explain women AGAIN? 💀"
- If they're venting: "Bro, she's not coming back 😤 Move on."
- Keep it SHORT - no long explanations

PREVIOUS CONTEXT:
{history}

USER SAYS: "{user_message}"

YOUR SAVAGE RESPONSE (1-2 sentences max):"""

    # Format conversation history
    history_text = ""
    if conversation_history:
        for msg in conversation_history[-4:]:  # Last 4 messages for context
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "user":
                history_text += f"User: {content}\n"
            elif role == "assistant":
                history_text += f"ManSlater: {content}\n"
    
    if not history_text:
        history_text = "[First message]"
    
    try:
        response = together_client.chat.completions.create(
            model=ROAST_MODEL,
            messages=[
                {"role": "system", "content": "You are ManSlater. Be savage, mocking, and brutally honest. Keep responses SHORT (1-2 sentences). Always include emoji."},
                {"role": "user", "content": chat_prompt.format(
                    history=history_text,
                    user_message=user_message
                )}
            ],
            max_tokens=50,
            temperature=0.9,
            top_p=0.95
        )
        
        savage_response = response.choices[0].message.content.strip()
        
        # Ensure it has emoji
        if not any(emoji in savage_response for emoji in ['💀', '🔥', '😐', '🙄', '😤']):
            savage_response += " 💀"
        
        # Keep it short (safety check)
        sentences = savage_response.split('.')
        if len(sentences) > 2:
            savage_response = '. '.join(sentences[:2]) + '.'
            
        return savage_response
        
    except Exception as e:
        print(f"Error generating chat response: {e}")
        return "Back again with problems? 💀 What happened now, genius?"

# ============ LANGGRAPH SETUP ============

class ConversationState(TypedDict):
    messages: Annotated[List, operator.add]

llm = ChatOpenAI(
    base_url="https://api.together.xyz/v1",
    api_key=TOGETHER_API_KEY,
    model=FINE_TUNED_MODEL,
    temperature=0.7,
    max_tokens=150
)

def response_node(state: ConversationState) -> ConversationState:
    """Route to TRANSLATION or CHAT mode based on intent"""
    messages = state.get("messages", [])
    
    # Get last user message
    user_message = ""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            user_message = msg.content
            break
    
    # Classify intent
    intent = classify_intent(user_message)
    print(f"🔍 Intent detected: {intent.upper()} for message: {user_message[:50]}...")
    
    # Convert history to dict format
    conversation_history = []
    for msg in messages[:-1]:
        if isinstance(msg, HumanMessage):
            conversation_history.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage):
            conversation_history.append({"role": "assistant", "content": msg.content})
    
    if intent == "translate":
        # TRANSLATION MODE: Roast + Advice
        roast = generate_roast(user_message)
        advice = generate_advice(user_message, conversation_history)
        combined_response = f"{roast}\n{advice}"
    else:
        # CHAT MODE: Just savage response (no advice format)
        combined_response = generate_savage_chat_response(user_message, conversation_history)
    
    return {"messages": [AIMessage(content=combined_response)]}

def create_advisor_graph():
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
        history = redis_manager.get_chat_history(self.session_id)
        messages = []
        for msg in history:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                messages.append(AIMessage(content=msg["content"]))
        return messages
    
    def send_message(self, user_message: str):
        redis_manager.save_chat_message(self.session_id, "user", user_message)
        
        history = self.load_history_from_redis()
        input_data = {"messages": history + [HumanMessage(content=user_message)]}
        
        result = self.graph.invoke(input_data, self.config)
        
        ai_response = None
        for msg in reversed(result["messages"]):
            if isinstance(msg, AIMessage):
                ai_response = msg.content
                break
        
        if not ai_response:
            ai_response = "Something wrong? 💀\nTalk to me."
        
        redis_manager.save_chat_message(self.session_id, "assistant", ai_response)
        
        return ai_response

# ============ FASTAPI APP ============

app = FastAPI(title="Manslater API", version="2.3.0")

# Ensure uploads directory exists and serve it
UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
        "https://manslater.in",
        "https://www.manslater.in",
        "https://zlz7jsh7-3000.inc1.devtunnels.ms"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============ MEMORY MANAGEMENT ============

active_instances = {}
_instance_access_times = {}  # Track last access time for cleanup

def cleanup_old_instances():
    """Remove instances that haven't been accessed recently"""
    global active_instances, _instance_access_times
    
    current_time = time.time()
    sessions_to_remove = []
    
    for session_id, last_access in list(_instance_access_times.items()):
        # Remove instances not accessed in last 30 minutes
        if current_time - last_access > 1800:  # 30 minutes
            sessions_to_remove.append(session_id)
    
    for session_id in sessions_to_remove:
        if session_id in active_instances:
            del active_instances[session_id]
        if session_id in _instance_access_times:
            del _instance_access_times[session_id]
    
    # If we're still over the limit, remove oldest instances
    if len(active_instances) > MAX_ACTIVE_INSTANCES:
        sorted_sessions = sorted(
            _instance_access_times.items(),
            key=lambda x: x[1]
        )
        to_remove = len(active_instances) - MAX_ACTIVE_INSTANCES
        for session_id, _ in sorted_sessions[:to_remove]:
            if session_id in active_instances:
                del active_instances[session_id]
            if session_id in _instance_access_times:
                del _instance_access_times[session_id]
    
    return len(sessions_to_remove)

# ============ MODELS ============

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    session_id: str
    rate_limit_info: dict
    intent: Optional[str] = None  # NEW: Show detected intent

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
    return {
        "message": "Welcome to Manslater API v2.3.0",
        "version": "2.3.0",
        "features": [
            "🧠 Intent Classification: Auto-detect translation vs chat",
            "💬 Translation Mode: Roast + 'just say' advice",
            "😤 Chat Mode: Savage banter responses",
            f"⏱️ Rate Limit: {CHAT_LIMIT_PER_DEVICE} chats per {RATE_LIMIT_TTL//3600}h"
        ],
        "modes": {
            "translate": "User asks what woman's words mean → Roast + Advice",
            "chat": "User just talking/venting → Savage ManSlater roast"
        }
    }

@app.get("/health")
@app.head("/health")
async def health_check():
    try:
        redis_manager.client.ping()
        redis_status = "connected"
    except:
        redis_status = "disconnected"
    
    return {
        "status": "healthy",
        "fine_tuned_model": FINE_TUNED_MODEL,
        "roast_model": ROAST_MODEL,
        "redis": redis_status,
        "features": ["intent_classification", "dual_mode_responses"]
    }

@app.get("/rate-limit", response_model=RateLimitInfo)
async def check_rate_limit(request: Request, user_agent: Optional[str] = Header(None)):
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
    Smart chat endpoint with intent detection
    - TRANSLATE mode: "She said..." → Roast + Advice
    - CHAT mode: "Hey bro" → Savage banter
    """
    try:
        # Global rate limiting
        await check_global_rate_limit()
        
        try:
            device_id = generate_device_id(request, user_agent)
            try:
                rate_info = redis_manager.check_rate_limit(device_id)
            except Exception as redis_err:
                # Graceful degradation if Redis is down
                print(f"Redis error in rate limit check: {redis_err}")
                rate_info = {"allowed": True, "current_usage": 0, "limit": CHAT_LIMIT_PER_DEVICE, "remaining": CHAT_LIMIT_PER_DEVICE}
            
            if not rate_info["allowed"]:
                try:
                    ttl = redis_manager.get_device_ttl(device_id)
                except:
                    ttl = RATE_LIMIT_TTL
                hours_remaining = ttl // 3600
                minutes_remaining = (ttl % 3600) // 60
                
                rate_limit_message = f"Rate limit hit 😐\nYou've used all {CHAT_LIMIT_PER_DEVICE} chats. Try again in {hours_remaining}h {minutes_remaining}m."
                
                session_id = request_body.session_id or f"session_{os.urandom(8).hex()}"
                try:
                    redis_manager.save_chat_message(session_id, "user", request_body.message)
                    redis_manager.save_chat_message(session_id, "assistant", rate_limit_message)
                except:
                    pass  # Continue even if Redis fails
                
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
            
            try:
                redis_manager.increment_request_count("chat")
            except:
                pass  # Continue even if Redis fails
            
            session_id = request_body.session_id or f"session_{os.urandom(8).hex()}"
            
            # Cleanup old instances periodically
            if len(active_instances) > MAX_ACTIVE_INSTANCES * 0.9:  # Cleanup at 90% capacity
                cleanup_old_instances()
            
            # Get or create instance with memory management
            if session_id not in active_instances:
                # Check limit before creating
                if len(active_instances) >= MAX_ACTIVE_INSTANCES:
                    cleanup_old_instances()
                active_instances[session_id] = Manslater(session_id)
            
            # Update access time
            _instance_access_times[session_id] = time.time()
            
            advisor = active_instances[session_id]
            
            # Get response (intent detection happens inside)
            response_text = advisor.send_message(request_body.message)
            
            # Detect intent for response metadata
            detected_intent = classify_intent(request_body.message)
            
            try:
                new_usage = redis_manager.increment_device_usage(device_id)
                updated_rate_info = redis_manager.check_rate_limit(device_id)
                ttl = redis_manager.get_device_ttl(device_id)
            except:
                updated_rate_info = rate_info
                ttl = RATE_LIMIT_TTL
            
            return ChatResponse(
                response=response_text,
                session_id=session_id,
                intent=detected_intent,  # NEW: Show what mode was used
                rate_limit_info={
                    "current_usage": updated_rate_info["current_usage"],
                    "limit": updated_rate_info["limit"],
                    "remaining": updated_rate_info["remaining"],
                    "reset_in_seconds": ttl
                }
            )
        finally:
            # Always release global rate limit
            await release_global_rate_limit()
    
    except HTTPException:
        await release_global_rate_limit()
        raise
    except Exception as e:
        await release_global_rate_limit()
        print(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=f"Chat error: {str(e)}")


@app.post('/upload')
async def upload_image(request: Request, file: UploadFile = File(...)):
    """Save uploaded image to uploads/ and return a public URL.

    Note: This is a minimal, unauthenticated upload for convenience. In
    production you'd add auth, rate-limiting, and remove long-term public
    access or use signed URLs.
    """
    try:
        # generate a safe filename
        import time, secrets
        ext = os.path.splitext(file.filename)[1] or ".png"
        fname = f"{int(time.time())}_{secrets.token_hex(8)}{ext}"
        dest = os.path.join(UPLOAD_DIR, fname)

        contents = await file.read()
        with open(dest, "wb") as f:
            f.write(contents)

        # Build public URL using request.base_url
        base = str(request.base_url).rstrip("/")
        public_url = f"{base}/uploads/{fname}"

        return {"url": public_url}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {e}")

@app.post("/translate", response_model=TranslateResponse)
async def translate_text(request: TranslateRequest):
    """Single-turn translation endpoint (NO rate limit)"""
    if not request.text:
        raise HTTPException(status_code=400, detail="No text provided")

    try:
        redis_manager.increment_request_count("translate")
        
        cached_translation = redis_manager.get_translation_cache(request.text)
        if cached_translation:
            return TranslateResponse(
                translatedText=cached_translation,
                cached=True
            )
        
        roast = generate_roast(request.text)
        advice = generate_advice(request.text)
        combined = f"{roast}\n{advice}"
        
        redis_manager.set_translation_cache(request.text, combined)
        
        return TranslateResponse(
            translatedText=combined,
            cached=False
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Translation failed: {str(e)}")

@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    deleted = redis_manager.delete_session(session_id)
    
    if session_id in active_instances:
        del active_instances[session_id]
    
    if deleted:
        return {"message": "Session deleted successfully"}
    raise HTTPException(status_code=404, detail="Session not found")

@app.get("/sessions")
async def list_sessions():
    sessions = redis_manager.get_all_sessions()
    return {
        "active_sessions": sessions,
        "total": len(sessions)
    }

@app.get("/analytics")
async def get_analytics():
    return redis_manager.get_analytics()

@app.delete("/cache/translations")
async def clear_translation_cache():
    try:
        pattern = "translation:*"
        keys = redis_manager.client.keys(pattern)
        if keys:
            redis_manager.client.delete(*keys)
        return {"message": f"Cleared {len(keys)} cached translations"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cache clear failed: {str(e)}")

@app.post("/admin/reset-device-limit")
async def reset_device_limit(
    request: Request,
    user_agent: Optional[str] = Header(None),
    admin_key: str = Header(None)
):
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
    print("🚀 Starting Manslater API v2.3.0")
    print("   🧠 Intent Classification Active")
    print("=" * 60)
    print("Server: http://localhost:8000")
    print("Docs: http://localhost:8000/docs")
    print("=" * 60)
    print("\n🎯 Response Modes:")
    print("  TRANSLATE: 'She said X' → Roast + Advice")
    print("  CHAT: 'Hey bro' → Savage ManSlater banter")
    print("=" * 60)
    print(f"\nModels:")
    print(f"  Intent Classifier: {ROAST_MODEL}")
    print(f"  Roast Generator: {ROAST_MODEL}")
    print(f"  Advice Model: {FINE_TUNED_MODEL}")
    print("=" * 60)
    
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)