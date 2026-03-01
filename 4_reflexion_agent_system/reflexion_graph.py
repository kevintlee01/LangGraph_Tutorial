from langchain_core.messages import ToolMessage, HumanMessage
from langgraph.graph import END, StateGraph, MessagesState
from chains import revisor_chain, first_responder_chain
from execute_tools import execute_tools

# Node wrapper functions to ensure dict output
def call_draft(state: MessagesState):
    res = first_responder_chain.invoke(state)
    return {"messages": [res]}

def call_revisor(state: MessagesState):
    res = revisor_chain.invoke(state)
    return {"messages": [res]}

builder = StateGraph(MessagesState)
MAX_ITERATIONS = 2

builder.add_node("draft", call_draft)
builder.add_node("execute_tools", execute_tools)
builder.add_node("revisor", call_revisor)

builder.add_edge("draft", "execute_tools")
builder.add_edge("execute_tools", "revisor")

# Conditional logic: returns keys for the mapping dict
def event_loop(state: MessagesState) -> str:
    messages = state["messages"]
    # Check how many ToolMessages are in history to count iterations
    tool_count = sum(1 for m in messages if isinstance(m, ToolMessage))
    
    if tool_count >= MAX_ITERATIONS:
        return "end"
    return "continue"

builder.add_conditional_edges(
    "revisor", 
    event_loop, 
    {
        "continue": "execute_tools", # This creates the loop back
        "end": END
    }
)

builder.set_entry_point("draft")
app = builder.compile()

print(app.get_graph().draw_mermaid())

# Test it
response = app.invoke({
    "messages": [HumanMessage(content="How can small businesses use AI to grow?")]
})

# Print the final answer found in the last tool call
print(response["messages"][-1].tool_calls[0]["args"]["answer"])