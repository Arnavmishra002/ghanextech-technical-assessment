"""Streamlit chat UI for the tool-calling agent."""

import json

import streamlit as st
from dotenv import load_dotenv

from agent.agent import Agent
from agent.llm import LLMError, get_api_key, make_client, model_name
from agent.tools import TOOLS

load_dotenv()

# One message can mean several LLM calls on a shared free key, so each
# browser session gets a fixed number of messages before it needs its own key.
MAX_MESSAGES_PER_SESSION = 20

EXAMPLES = [
    "Hi! What can you do?",
    "What's the weather in Pune right now?",
    "Convert 250 USD to INR",
    "What is 18% GST on 1249.50?",
    "Who is Satya Nadella?",
    "Is it warmer in Delhi or Mumbai right now?",
]

st.set_page_config(page_title="Tool-calling agent", layout="centered")

state = st.session_state
state.setdefault("messages", [])  # {"role", "content", "tool_calls"}
state.setdefault("sent", 0)

st.title("Tool-calling agent")
st.caption("A chat assistant that decides on its own when to use a calculator, live weather, Wikipedia or currency rates.")

with st.sidebar:
    st.header("Tools")
    for tool in TOOLS.values():
        st.markdown(f"**{tool.name}**: {tool.description.split('.')[0]}.")
    st.divider()
    st.header("Try an example")
    for example in EXAMPLES:
        if st.button(example, width="stretch"):
            state.pending = example
    st.divider()
    user_key = st.text_input(
        "Your Groq API key (optional)",
        type="password",
        help="Leave empty to use the demo key. A key typed here stays in your browser session only.",
    )
    st.caption(f"Model: {model_name()}")
    if st.button("Clear chat"):
        state.messages = []


def show_tool_calls(calls):
    for call in calls:
        failed = "error" in call["result"]
        label = f"{'Tool failed' if failed else 'Used tool'}: {call['name']}"
        with st.expander(label):
            st.caption("Arguments")
            st.code(call["arguments"], language="json")
            st.caption("Result")
            st.code(json.dumps(call["result"], indent=2, ensure_ascii=False), language="json")


for message in state.messages:
    with st.chat_message(message["role"]):
        if message.get("tool_calls"):
            show_tool_calls(message["tool_calls"])
        st.markdown(message["content"])

prompt = st.chat_input("Ask me anything") or state.pop("pending", None)

if prompt:
    history = [{"role": m["role"], "content": m["content"]} for m in state.messages]
    state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        calls = []
        if state.sent >= MAX_MESSAGES_PER_SESSION and not user_key:
            text = (
                f"This demo allows {MAX_MESSAGES_PER_SESSION} messages per session on the shared key. "
                "Add your own free Groq API key in the sidebar to keep going."
            )
        elif not (user_key or get_api_key()):
            text = "No API key is configured. Add your Groq API key in the sidebar."
        else:
            try:
                agent = Agent(make_client(user_key or None))
                with st.spinner("Thinking..."):
                    reply = agent.run(history, prompt)
                state.sent += 1
                text = reply.text
                calls = [{"name": c.name, "arguments": c.arguments, "result": c.result} for c in reply.tool_calls]
            except LLMError as exc:
                text = f"Sorry, something went wrong: {exc}"
        show_tool_calls(calls)
        st.markdown(text)
    state.messages.append({"role": "assistant", "content": text, "tool_calls": calls})
