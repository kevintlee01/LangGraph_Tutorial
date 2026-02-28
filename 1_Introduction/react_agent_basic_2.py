from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_tavily import TavilySearch
import datetime

@tool
def get_system_time(format: str = "%Y-%m-%d %H:%M:%S") -> str:
    """
    Returns the current system date and time as a string. 
    Use this when the user asks for 'today', 'now', or current time/date.
    """
    
    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime(format)
    return formatted_time

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-lite")

search_tool = TavilySearch(max_results=3, search_depth="basic")

tools = [search_tool, get_system_time]

agent = create_agent(tools=tools, model=llm)

query = "Give me today's weather in San Jose and how may days is it till Christmas?"
response = agent.invoke({"messages": [("human", query)]})

final_content = response["messages"][-1].content

if isinstance(final_content, list):
    print(final_content[0].get("text"))
else:
    print(final_content)