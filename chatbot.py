import streamlit as st
from langchain_core.messages.ai import AIMessage
from langchain_core.messages import HumanMessage
from Serenity import invoke_our_graph

st.title("Serenity Chatbot")
# This is the message with which the system opens the conversation.
WELCOME_MSG = "Welcome to SERENITY. How can I help you today?"
st.chat_message("assistant").write(WELCOME_MSG)
if "messages" not in st.session_state:
    st.session_state["messages"] = []

for msg in st.session_state.messages:
    if type(msg) == AIMessage:
        st.chat_message("assistant").write(msg.content)
    if type(msg) == HumanMessage:
        st.chat_message("user").write(msg.content)

# takes new input in chat box from user and invokes the graph
if prompt := st.chat_input():
    st.session_state.messages.append(HumanMessage(content=prompt))
    st.chat_message("user").write(prompt)
    response = invoke_our_graph(st.session_state.messages)
    # Process the AI's response and handles graph events using the callback mechanism
    st.chat_message("assistant").write(response["messages"][-1].content)
    st.session_state.messages.append(AIMessage(content=response["messages"][-1].content))