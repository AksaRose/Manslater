from typing import TypedDict, Annotated, List
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import operator
from dotenv import load_dotenv
import os

load_dotenv(dotenv_path=os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"))
os.environ["OPENAI_API_KEY"] = os.environ.get("TOGETHER_API_KEY")

# 1. DEFINE STATE
class ConversationState(TypedDict):
    """State that gets passed between nodes"""
    messages: Annotated[List, operator.add]  
    what_she_said: str  
    tone: str  
    context: str  
    recent_events: str
    relationship_length: str
    his_last_action: str
    info_complete: bool  
    questions_asked: int


# 2. INITIALIZE LLMs
# Fine-tuned model for translations
translation_llm = ChatOpenAI(
    base_url="https://api.together.xyz/v1",
    api_key=os.environ.get("TOGETHER_API_KEY"),
    model="rahmanhabeeb360_24ae/Meta-Llama-3.1-8B-Instruct-Reference-ft-manslaterfinetune-417bfc68",
    temperature=0.8,
    max_tokens=300
)

# Larger model for better conversational questions
question_llm = ChatOpenAI(
    base_url="https://api.together.xyz/v1",
    api_key=os.environ.get("TOGETHER_API_KEY"),
    model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",  # More capable model
    temperature=0.9,
    max_tokens=150
)


# 3. NODE FUNCTIONS
def parse_input_node(state: ConversationState) -> ConversationState:
    """Extract what she said from user's first message"""
    last_message = None
    for m in reversed(state["messages"]):
        if isinstance(m, HumanMessage):
            last_message = m.content
            break

    if not last_message:
        return {}
    
    # Only set what_she_said if it's empty
    if not state.get("what_she_said"):
        return {"what_she_said": last_message}
    
    return {}


def assessment_node(state: ConversationState) -> ConversationState:
    """Check if we have enough information to give advice"""
    info_fields = [
        state.get("what_she_said"),
        state.get("tone"),
        state.get("context"),
        state.get("recent_events"),
        state.get("relationship_length"),
        state.get("his_last_action")
    ]
    
    filled_fields = sum(1 for field in info_fields if field)
    
    # Need at least 4 pieces of info or asked 6 questions
    if filled_fields >= 4 or state.get("questions_asked", 0) >= 6:
        return {"info_complete": True}
    
    return {}


def probing_node(state: ConversationState) -> ConversationState:
    """Ask targeted, conversational questions to gather information"""
    questions_asked = state.get("questions_asked", 0)
    
    # Build recent conversation context
    recent_messages = state["messages"][-4:] if len(state["messages"]) > 4 else state["messages"]
    conversation_context = "\n".join([
        f"{'User' if isinstance(m, HumanMessage) else 'You'}: {m.content}" 
        for m in recent_messages
    ])
    
    # Determine what to ask based on missing info
    if not state.get("tone"):
        system_prompt = f"""You are a brutally honest relationship advisor for guys. You speak like a bro giving tough love.

Recent chat:
{conversation_context}

She said: "{state.get('what_she_said')}"

Ask about the TONE/VIBE in a conversational, snark and savage mode. Don't make it feel like an interview.
Examples of good questions:
- "Okay genius, how did she say it? Ice-cold mode or annoyed-but-trying-not-to-explode? Use your brain."
- "Bro, give me the vibe. Calm like ‘I don’t care’ or was there an edge to it?
- "Hold up - was this a cold 'fine' or a sarcastic 'fine'? Huge difference, buddy."

Keep it short, direct, and conversational. ONE question only."""
        
    elif not state.get("context"):
        system_prompt = f"""You are a brutally honest relationship advisor for guys. You speak like a bro giving tough love.

Recent chat:
{conversation_context}

She said: "{state.get('what_she_said')}" with a {state.get('tone', 'certain')} tone.

Ask about CONTEXT in a natural, flowing way but snark, savage mode. What was happening right before?
Examples of good questions:
- "Alright, so what were you guys doing when she dropped that bomb?"
- "Context time - what happened like 5 minutes before she said this?"
- "Walk me through it. What led up to this moment?"

Keep it conversational and brief. ONE question only."""
        
    elif not state.get("recent_events"):
        system_prompt = f"""You are a brutally honest relationship advisor for guys. You speak like a bro giving tough love.

Recent chat:
{conversation_context}

Ask about recent events/fights in a natural, snark, savage way.
Examples of good questions:
- "Real talk - did you maybe forget something important recently? Anniversary? Plans?"
- "Have you guys been fighting about anything the past couple days?"
- "Did something go down between you two that you might've brushed off?"

Keep it conversational. ONE question only."""

    elif not state.get("relationship_length"):
        system_prompt = f"""You are a brutally honest relationship advisor for guys. You speak like a bro giving tough love.

Recent chat:
{conversation_context}

Ask about the relationship length/history naturally.
Examples:
- "How long have you two been together anyway?"
- "Is this a new thing or have you been together a while?"
- "Quick question - how long you been with her?"

Keep it brief and natural. ONE question only."""

    elif not state.get("his_last_action"):
        system_prompt = f"""You are a brutally honest relationship advisor for guys. You speak like a bro giving tough love.

Recent chat:
{conversation_context}

Ask what HE did or said right before her response.
Examples:
- "What did YOU say or do right before she hit you with that?"
- "Okay but what were you doing before this? That's key here."
- "Hold on - what did you say that made her respond like that?"

Keep it conversational. ONE question only."""
        
    else:
        system_prompt = f"""You are a brutally honest relationship advisor for guys. You speak like a bro giving tough love.

Recent chat:
{conversation_context}

Ask one more clarifying question to understand the situation better. 
Be natural, savage, and snarky and conversational. Keep it brief."""
    
    messages = [
        SystemMessage(content=system_prompt)
    ]
    
    # Use the larger model for better questions
    response = question_llm.invoke(messages)
    
    return {
        "messages": [AIMessage(content=response.content)],
        "questions_asked": questions_asked + 1
    }


def extract_info_node(state: ConversationState) -> ConversationState:
    """Extract structured info from user's response"""
    last_user_message = None
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            last_user_message = msg.content
            break
    
    if not last_user_message:
        return {}
    
    # Skip extraction if this is the first message (what_she_said)
    if state.get("questions_asked", 0) == 0:
        return {}
    
    # Extract based on what's missing
    if not state.get("tone"):
        return {"tone": last_user_message}
    elif not state.get("context"):
        return {"context": last_user_message}
    elif not state.get("recent_events"):
        return {"recent_events": last_user_message}
    elif not state.get("relationship_length"):
        return {"relationship_length": last_user_message}
    elif not state.get("his_last_action"):
        return {"his_last_action": last_user_message}
    
    return {}


def translation_node(state: ConversationState) -> ConversationState:
    """Give final translation and advice"""
    system_prompt = f"""You're a brutally honest relationship advisor. Based on this info:

- She said: "{state.get('what_she_said', 'N/A')}"
- Tone: {state.get('tone', 'N/A')}
- Context: {state.get('context', 'N/A')}
- Recent events: {state.get('recent_events', 'N/A')}
- Relationship length: {state.get('relationship_length', 'N/A')}
- What he did: {state.get('his_last_action', 'N/A')}

Give advice in EXACTLY this format:

Translation: [What she really means - be direct and insightful]

Say: "[Exact phrase he should use - word for word]"

Action: [1-2 quick action steps]

Be bro-like but helpful. Keep it real."""

    messages = [
        SystemMessage(content=system_prompt),
        *state["messages"][-3:]
    ]
    
    # Use fine-tuned model for translation
    response = translation_llm.invoke(messages)
    
    return {
        "messages": [AIMessage(content=response.content)]
    }


# 4. ROUTING LOGIC
def should_continue_probing(state: ConversationState) -> str:
    """Decide whether to probe more or give advice"""
    if state.get("info_complete"):
        return "translate"
    return "probe"


# 5. BUILD THE GRAPH
def create_advisor_graph():
    """Build and compile the conversation graph"""
    workflow = StateGraph(ConversationState)
    
    workflow.add_node("parse_input", parse_input_node)
    workflow.add_node("extract_info", extract_info_node)
    workflow.add_node("assess", assessment_node)
    workflow.add_node("probe", probing_node)
    workflow.add_node("translate", translation_node)
    
    workflow.set_entry_point("parse_input")
    
    workflow.add_edge("parse_input", "extract_info")
    workflow.add_edge("extract_info", "assess")
    
    workflow.add_conditional_edges(
        "assess",
        should_continue_probing,
        {
            "probe": "probe",
            "translate": "translate"
        }
    )
    
    workflow.add_edge("probe", END)
    workflow.add_edge("translate", END)
    
    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)
    return app


# 6. RUN THE ADVISOR
class Manslater:
    def __init__(self):
        self.graph = create_advisor_graph()
        self.thread_id = "conversation_" + str(os.urandom(4).hex())
        self.config = {"configurable": {"thread_id": self.thread_id}}
    
    def send_message(self, user_message: str):
        """Send user message and get response"""
        # Add user message
        input_data = {
            "messages": [HumanMessage(content=user_message)]
        }
        
        # Invoke graph with checkpointing
        result = self.graph.invoke(input_data, self.config)
        
        # Return last AI message
        for msg in reversed(result["messages"]):
            if isinstance(msg, AIMessage):
                return msg.content
        
        return "Something went wrong, bro."
    
    def get_state(self):
        """Get current conversation state (for debugging)"""
        return self.graph.get_state(self.config)


# 7. USAGE
if __name__ == "__main__":
    advisor = Manslater()
    
    print("=== Girlfriend Translator (LangGraph Edition) ===\n")
    print("AI: Alright genius, what did she say that's got you panicking?\n")
    
    while True:
        user_input = input("You: ")
        
        if user_input.lower() in ['quit', 'exit', 'bye']:
            print("AI: Good luck out there, champ! 💪")
            break
        
        response = advisor.send_message(user_input)
        print(f"\nAI: {response}\n")