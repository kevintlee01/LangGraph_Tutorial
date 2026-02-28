from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import HarmCategory, HarmBlockThreshold

generation_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an influencer agent tasked with writing excellent Twitter posts. "
            "Generate the best possible Twitter post based on the user's input. "
            "If the user provides a critique, respond with a revised version of your previous attempts."
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

reflection_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a viral Twitter influencer grading a Tweet. "
            "Generate a critique of the Tweet and recommendations for improvement. "
            "Always provide detailed recommendations including request for length, virality, style, etc."
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)


llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    safety_settings={
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    }
)

generation_chain = generation_prompt | llm
reflection_chain = reflection_prompt | llm