import json
from types import SimpleNamespace

import pytest

from agent import agent as agent_module
from agent.agent import MAX_HISTORY_MESSAGES, Agent
from agent.llm import LLMError


def text_message(content):
    return SimpleNamespace(content=content, tool_calls=None)


def tool_message(*calls):
    return SimpleNamespace(
        content="",
        tool_calls=[
            SimpleNamespace(id=f"call_{i}", function=SimpleNamespace(name=name, arguments=json.dumps(args)))
            for i, (name, args) in enumerate(calls)
        ],
    )


@pytest.fixture
def scripted(monkeypatch):
    """Replace the model with a list of scripted replies; record what it was sent."""
    sent = []

    def install(replies):
        replies = list(replies)

        def fake_complete(client, messages, tools):
            sent.append([dict(m) for m in messages])
            reply = replies.pop(0)
            if isinstance(reply, Exception):
                raise reply
            return reply

        monkeypatch.setattr(agent_module, "complete", fake_complete)
        return sent

    return install


def test_greeting_is_answered_without_tools(scripted):
    sent = scripted([text_message("Hello! How can I help?")])
    reply = Agent(client=None).run([], "hi")
    assert reply.text == "Hello! How can I help?"
    assert reply.tool_calls == []
    assert len(sent) == 1


def test_tool_result_is_sent_back_to_model(scripted):
    sent = scripted([tool_message(("calculator", {"expression": "1249.5 * 0.18"})), text_message("GST is 224.91.")])
    reply = Agent(client=None).run([], "18% GST on 1249.50?")
    assert reply.text == "GST is 224.91."
    assert [c.name for c in reply.tool_calls] == ["calculator"]
    assert reply.tool_calls[0].result["result"] == pytest.approx(224.91)

    second_call = sent[1]
    assert second_call[-2]["role"] == "assistant" and second_call[-2]["tool_calls"][0]["id"] == "call_0"
    assert second_call[-1]["role"] == "tool" and second_call[-1]["tool_call_id"] == "call_0"
    assert json.loads(second_call[-1]["content"])["result"] == pytest.approx(224.91)


def test_several_tools_in_sequence(scripted):
    scripted(
        [
            tool_message(("calculator", {"expression": "2+2"}), ("calculator", {"expression": "3*3"})),
            tool_message(("calculator", {"expression": "4+9"})),
            text_message("Total is 13."),
        ]
    )
    reply = Agent(client=None).run([], "add them")
    assert [c.result["result"] for c in reply.tool_calls] == [4, 9, 13]
    assert reply.text == "Total is 13."


def test_tool_error_goes_to_model_not_to_crash(scripted):
    sent = scripted([tool_message(("calculator", {"expression": "1/0"})), text_message("You cannot divide by zero.")])
    reply = Agent(client=None).run([], "1/0?")
    assert reply.tool_calls[0].result == {"error": "Division by zero."}
    assert "Division by zero" in sent[1][-1]["content"]
    assert reply.text == "You cannot divide by zero."


def test_stops_after_max_rounds(scripted):
    scripted([tool_message(("calculator", {"expression": "1+1"}))] * 3)
    reply = Agent(client=None, max_rounds=3).run([], "loop forever")
    assert "too many tool steps" in reply.text
    assert len(reply.tool_calls) == 3


def test_history_is_kept_and_trimmed(scripted):
    sent = scripted([text_message("ok")])
    history = [{"role": "user" if i % 2 == 0 else "assistant", "content": f"m{i}"} for i in range(30)]
    Agent(client=None).run(history, "latest")
    messages = sent[0]
    assert messages[0]["role"] == "system"
    assert len(messages) == 1 + MAX_HISTORY_MESSAGES + 1
    assert messages[1]["content"] == "m10" and messages[-1]["content"] == "latest"


def test_empty_model_reply_gets_fallback_text(scripted):
    scripted([text_message("")])
    assert "could not come up with an answer" in Agent(client=None).run([], "?").text


def test_llm_errors_propagate_for_the_ui_to_show(scripted):
    scripted([LLMError("rate limit")])
    with pytest.raises(LLMError):
        Agent(client=None).run([], "hi")
