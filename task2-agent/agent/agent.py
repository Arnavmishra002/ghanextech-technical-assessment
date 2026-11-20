"""The agent loop.

1. Send the conversation plus the tool schemas to the model.
2. If the model asks for tools, run them and send the results back.
3. Repeat until the model answers in plain text, or the step limit is hit.

The model decides when to use a tool (tool_choice="auto"). The code only
runs what was asked for and turns every failure into text the model can
explain, so one bad tool call never crashes the chat.
"""

import json
from dataclasses import dataclass, field
from datetime import date

from .llm import complete
from .tools import run_tool, tool_schemas

MAX_TOOL_ROUNDS = 5
MAX_HISTORY_MESSAGES = 20  # last 10 exchanges; keeps each request small

SYSTEM_PROMPT = """You are a helpful assistant in a chat app. Today is {today}.

You can call tools. Use one only when it is really needed:
- calculator: for any arithmetic beyond trivial mental maths. Never do long calculations in your head.
- get_weather: for current weather or today's forecast in a place.
- search_wikipedia: for facts about people, places, events or topics you are unsure about or that may have changed.
- convert_currency: for converting money between currencies.

Answer greetings, small talk and general questions directly, without tools.
If a question needs several tools, call them one after another (for example weather in two cities, then compare).
If a tool returns an error, explain the problem in simple words and suggest what the user can try. Never make up tool results.
Keep answers short, clear and friendly."""


@dataclass
class ToolCallRecord:
    name: str
    arguments: str
    result: dict


@dataclass
class AgentReply:
    text: str
    tool_calls: list[ToolCallRecord] = field(default_factory=list)


class Agent:
    def __init__(self, client, max_rounds: int = MAX_TOOL_ROUNDS):
        self.client = client
        self.max_rounds = max_rounds

    def run(self, history: list[dict], user_message: str) -> AgentReply:
        """history holds earlier {"role", "content"} turns, oldest first.

        Raises LLMError if the model itself cannot be reached.
        """
        messages = [{"role": "system", "content": SYSTEM_PROMPT.format(today=date.today().isoformat())}]
        messages += history[-MAX_HISTORY_MESSAGES:]
        messages.append({"role": "user", "content": user_message})
        records: list[ToolCallRecord] = []

        for _ in range(self.max_rounds):
            message = complete(self.client, messages, tool_schemas())
            if not message.tool_calls:
                text = (message.content or "").strip()
                return AgentReply(text or "Sorry, I could not come up with an answer. Please try again.", records)

            messages.append(
                {
                    "role": "assistant",
                    "content": message.content or "",
                    "tool_calls": [
                        {
                            "id": call.id,
                            "type": "function",
                            "function": {"name": call.function.name, "arguments": call.function.arguments},
                        }
                        for call in message.tool_calls
                    ],
                }
            )
            for call in message.tool_calls:
                result = run_tool(call.function.name, call.function.arguments)
                records.append(ToolCallRecord(call.function.name, call.function.arguments, result))
                messages.append(
                    {"role": "tool", "tool_call_id": call.id, "content": json.dumps(result, ensure_ascii=False)}
                )

        return AgentReply(
            "I stopped because this needed too many tool steps. Please ask in smaller parts.", records
        )
