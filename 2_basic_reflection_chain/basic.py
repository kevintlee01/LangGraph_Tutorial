from typing import List, Sequence
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import END, StateGraph, MessagesState

load_dotenv()

from chains import generation_chain, reflection_chain

graph = StateGraph(MessagesState)

REFLECT = "reflect"
GENERATE = "generate"

def generate_node(state: MessagesState):
    response = generation_chain.invoke(state)
    return {"messages": [response]}

def reflect_node(state: MessagesState):
    response = reflection_chain.invoke(state)
    return {"messages": [HumanMessage(content=response.content)]}

graph.add_node(GENERATE, generate_node)
graph.add_node(REFLECT, reflect_node)

graph.set_entry_point(GENERATE)

def should_continue(state):
    if len(state["messages"]) > 1: #adjust this value to control number of iterations
        return END
    
    return REFLECT

graph.add_conditional_edges(
    GENERATE, 
    should_continue, 
    {
        REFLECT: REFLECT,
        END: END
    }
)

graph.add_edge(REFLECT, GENERATE)

app = graph.compile()

# print(app.get_graph().draw_mermaid())
# print(app.get_graph().draw_ascii())

inputs = {"messages": [HumanMessage(content="AI Agents taking over content creation")]}
response = app.invoke(inputs)

for message in response["messages"]:
    print(f"{message.type.upper()}: {message.content}\n")