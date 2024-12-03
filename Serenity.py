from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
from langgraph.graph import StateGraph, START, END
from pprint import pprint
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
import streamlit as st

load_dotenv()
llm = ChatGroq(
    model="llama-3.1-70b-versatile",
    temperature=0,
    groq_api_key = os.getenv("GROQ_API_KEY")
)

class OrderState(TypedDict):
    """State representing the customer's order conversation."""

    messages: Annotated[list, add_messages]
    finished: bool


# The system instruction defines how the chatbot is expected to behave and includes
# rules for when to call different functions, as well as rules for the conversation, such
# as tone and what is permitted for discussion.
SERENITY_SYSINT = (
    "system"
    "You are a chatbot named SERENITY, an interactive mental health support assistant."
    "You have been developed by a great team of mental health professionals and developers in a company named CherSolutions, "
    "and one of them is the famous Ghassen Cherif who is a great AI researcher and a mental health advocate. " 
    "A human will talk to you about his problems and they can be mental like depression, anxiety, stress... and you will provide"
    " support and advice. He can also ask you questions about mental health and you will provide information and resources. "
    "You can ask him questions to better understand his situation and provide better support. "
    "When you provide advices, you should make sure every time to provide a disclaimer that you are not a professional and that"
    "the user should consult a professional for a proper diagnosis and treatment. most of our users will be from Tunisia, so you have"
    " to provide resources and information that are relevant to Tunisia."
)




def chatbot_with_welcome_msg(state: OrderState) -> OrderState:
    response = llm.invoke([SERENITY_SYSINT] + state["messages"])
    return {"messages": [response]}


graph_builder = StateGraph(OrderState)
graph_builder.add_node("chatbot", chatbot_with_welcome_msg)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)
chatbot = graph_builder.compile()

def invoke_our_graph(st_messages):
    return chatbot.invoke({"messages": st_messages})