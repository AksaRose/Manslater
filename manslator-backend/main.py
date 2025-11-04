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


# 2. INITIALIZE LLM
llm = ChatOpenAI(
    base_url="https://api.together.xyz/v1",
    api_key=os.environ.get("TOGETHER_API_KEY"),
    model="rahmanhabeeb360_24ae/Meta-Llama-3.1-8B-Instruct-Reference-ft-manslater-4c03ddd8",
    temperature=0.8,
    max_tokens=100
)


# 3. NODE FUNCTIONS
def response_node(state: ConversationState) -> ConversationState:
    """Generate response using fine-tuned model"""
    
    messages = state.get("messages", [])
    
    system_prompt = """"You are a brutally honest relationship advisor for guys. "
                        "You speak like a bro giving tough love. "
                        "to understand the situation before giving advice. Never sugarcoat. "
                        "Always give EXACT phrases to say."
                        "Always be caring towards women"""
    
    llm_messages = [SystemMessage(content=system_prompt)] + messages[-8:]
    response = llm.invoke(llm_messages)
    
    return {
        "messages": [AIMessage(content=response.content)]
    }


# 4. BUILD THE GRAPH
def create_advisor_graph():
    """Build and compile the conversation graph"""
    workflow = StateGraph(ConversationState)
    
    workflow.add_node("response", response_node)
    
    workflow.set_entry_point("response")
    workflow.add_edge("response", END)
    
    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)
    return app


# 5. RUN THE ADVISOR
class Manslater:
    def __init__(self):
        self.graph = create_advisor_graph()
        self.thread_id = "conversation_" + str(os.urandom(4).hex())
        self.config = {"configurable": {"thread_id": self.thread_id}}
    
    def send_message(self, user_message: str):
        """Send user message and get response"""
        input_data = {
            "messages": [HumanMessage(content=user_message)]
        }
        
        result = self.graph.invoke(input_data, self.config)
        
        # Return last AI message
        for msg in reversed(result["messages"]):
            if isinstance(msg, AIMessage):
                return msg.content
        
        return "Something went wrong, bro."
    
    def get_state(self):
        """Get current conversation state (for debugging)"""
        return self.graph.get_state(self.config)


# 6. USAGE
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