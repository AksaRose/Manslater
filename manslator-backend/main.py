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
    info_complete: bool  
    questions_asked: int


# 2. INITIALIZE LLM
llm = ChatOpenAI(
    base_url="https://api.together.xyz/v1",
    api_key=os.environ.get("TOGETHER_API_KEY"),
    model="rahmanhabeeb360_24ae/Meta-Llama-3.1-8B-Instruct-Reference-ft-manslaterfinetune-417bfc68",
    temperature=0.8,
    max_tokens=200
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
        state.get("recent_events")
    ]
    
    filled_fields = sum(1 for field in info_fields if field)
    
    if filled_fields >= 3 or state.get("questions_asked", 0) >= 4:
        return {"info_complete": True}
    
    return {}


def probing_node(state: ConversationState) -> ConversationState:
    """Ask targeted questions to gather information"""
    questions_asked = state.get("questions_asked", 0)
    
    if not state.get("tone"):
        system_prompt = """You're a relationship advisor. Ask about the TONE/VIBE of how she said it. 
        Was she calm, cold, sarcastic, playful? Ask in a direct, bro-like way. ONE question only."""
        
    elif not state.get("context"):
        system_prompt = """You're a relationship advisor. Ask about CONTEXT - what happened right before she said this? 
        What were they doing? Ask directly. ONE question only."""
        
    elif not state.get("recent_events"):
        system_prompt = """You're a relationship advisor. Ask if there were any recent fights, 
        forgotten events, or screw-ups in the last 48 hours. Be direct. ONE question only."""
    else:
        system_prompt = """Ask one clarifying question about the situation. Be brief and direct."""
    
    messages = [
        SystemMessage(content=system_prompt),
        *state["messages"][-3:]
    ]
    
    response = llm.invoke(messages)
    
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
    
    if not state.get("tone"):
        return {"tone": last_user_message}
    elif not state.get("context"):
        return {"context": last_user_message}
    elif not state.get("recent_events"):
        return {"recent_events": last_user_message}
    
    return {}


def translation_node(state: ConversationState) -> ConversationState:
    """Give final translation and advice"""
    system_prompt = f"""You're a brutally honest relationship advisor. Based on this info:

- She said: "{state.get('what_she_said', 'N/A')}"
- Tone: {state.get('tone', 'N/A')}
- Context: {state.get('context', 'N/A')}
- Recent events: {state.get('recent_events', 'N/A')}

Give advice in EXACTLY this format:

Translation: [What she really means - be direct and insightful]

Say: "[Exact phrase he should use - word for word]"

Then add 1-2 lines of quick action steps if needed. Be bro-like but helpful."""

    messages = [
        SystemMessage(content=system_prompt),
        *state["messages"][-2:]
    ]
    
    response = llm.invoke(messages)
    
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