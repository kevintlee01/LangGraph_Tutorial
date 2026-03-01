import json
from typing import Dict, Any
from langchain_core.messages import AIMessage, ToolMessage
from langchain_tavily import TavilySearch
from langgraph.graph import MessagesState

tavily_tool = TavilySearch(max_results=2)

def execute_tools(state: MessagesState):
    messages = state["messages"]
    last_ai_message: AIMessage = messages[-1]
    
    if not hasattr(last_ai_message, "tool_calls") or not last_ai_message.tool_calls:
        return {"messages": []}
    
    tool_messages = []
    for tool_call in last_ai_message.tool_calls:
        if tool_call["name"] in ["AnswerQuestion", "ReviseAnswer"]:
            call_id = tool_call["id"]
            search_queries = tool_call["args"].get("search_queries", [])
            
            query_results = {}
            for query in search_queries:
                # The .invoke() method works exactly the same
                result = tavily_tool.invoke(query)
                query_results[query] = result
            
            tool_messages.append(
                ToolMessage(
                    content=json.dumps(query_results),
                    tool_call_id=call_id
                )
            )
    return {"messages": tool_messages}