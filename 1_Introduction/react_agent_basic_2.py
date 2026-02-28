from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain.agents import create_agent 
from langchain_tavily import TavilySearch

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

search_tool = TavilySearch(max_results=3, search_depth="basic")

tools = [search_tool]

agent = create_agent(tools=tools, model=llm)

query = "Give me today's weather in San Jose"
response = agent.invoke({"messages": [("human", query)]})

final_content = response["messages"][-1].content

if isinstance(final_content, list):
    print(final_content[0].get("text"))
else:
    print(final_content)