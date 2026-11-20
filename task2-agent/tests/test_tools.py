import json

import pytest
import requests

from agent.tools import TOOLS, ToolError, calculator, convert_currency, run_tool, tool_schemas
from conftest import FakeResponse

# ------------------------------------------------------------ calculator


@pytest.mark.parametrize(
    "expression, expected",
    [
        ("2 + 3 * 4", 14),
        ("(2 + 3) * 4", 20),
        ("1249.50 * 0.18", pytest.approx(224.91)),
        ("2 ^ 10", 1024),
        ("sqrt(144)", 12),
        ("-5 + abs(-3)", -2),
        ("round(pi, 3)", pytest.approx(3.142)),
        ("10 // 3 + 10 % 3", 4),
    ],
)
def test_calculator_evaluates(expression, expected):
    assert calculator(expression)["result"] == expected


@pytest.mark.parametrize(
    "expression",
    [
        "__import__('os').system('ls')",
        "().__class__",
        "open('secrets.txt')",
        "x + 1",
        "True + 1",
        "'a' * 3",
    ],
)
def test_calculator_blocks_anything_but_maths(expression):
    with pytest.raises(ToolError):
        calculator(expression)


def test_calculator_errors_are_clear():
    with pytest.raises(ToolError, match="Division by zero"):
        calculator("1 / 0")
    with pytest.raises(ToolError, match="Exponent too large"):
        calculator("9 ** 9 ** 9")
    with pytest.raises(ToolError, match="not a valid expression"):
        calculator("2 +* 3")
    with pytest.raises(ToolError, match="too long"):
        calculator("1+" * 150 + "1")
    with pytest.raises(ToolError, match="Math error"):
        calculator("sqrt(-1)")


# --------------------------------------------------------------- weather

GEO_PUNE = {"results": [{"name": "Pune", "admin1": "Maharashtra", "country": "India", "latitude": 18.5, "longitude": 73.9}]}
FORECAST = {
    "current": {
        "time": "2026-11-20T17:45",
        "temperature_2m": 26.8,
        "apparent_temperature": 24.2,
        "relative_humidity_2m": 23,
        "wind_speed_10m": 7.6,
        "weather_code": 0,
    },
    "daily": {"temperature_2m_max": [30.1], "temperature_2m_min": [14.2], "precipitation_probability_max": [0]},
}


def test_weather_success(fake_http):
    calls = fake_http({"geocoding-api": FakeResponse(200, GEO_PUNE), "api.open-meteo.com": FakeResponse(200, FORECAST)})
    out = json.loads(json.dumps(run_tool("get_weather", '{"location": "Pune"}')))
    assert out["place"] == "Pune, Maharashtra, India"
    assert out["condition"] == "clear sky"
    assert out["temperature_c"] == 26.8 and out["today_max_c"] == 30.1
    assert calls[1][1]["latitude"] == 18.5


def test_weather_unknown_place(fake_http):
    fake_http({"geocoding-api": FakeResponse(200, {"generationtime_ms": 0.1})})
    assert "Could not find a place" in run_tool("get_weather", '{"location": "Xyzzyville"}')["error"]


def test_weather_timeout(fake_http):
    fake_http({"geocoding-api": requests.Timeout()})
    assert "too long" in run_tool("get_weather", '{"location": "Pune"}')["error"]


def test_weather_server_error(fake_http):
    fake_http({"geocoding-api": FakeResponse(200, GEO_PUNE), "api.open-meteo.com": FakeResponse(503, {})})
    assert "HTTP 503" in run_tool("get_weather", '{"location": "Pune"}')["error"]


# ------------------------------------------------------------- wikipedia


def test_wikipedia_success_and_title_encoding(fake_http):
    calls = fake_http(
        {
            "w/api.php": FakeResponse(200, {"query": {"search": [{"title": "AC/DC"}, {"title": "AC/DC discography"}]}}),
            "page/summary": FakeResponse(
                200,
                {"title": "AC/DC", "extract": "AC/DC are an Australian rock band.", "type": "standard",
                 "content_urls": {"desktop": {"page": "https://en.wikipedia.org/wiki/AC/DC"}}},
            ),
        }
    )
    out = run_tool("search_wikipedia", '{"query": "acdc band"}')
    assert out["title"] == "AC/DC" and out["summary"].startswith("AC/DC are")
    assert out["other_matches"] == ["AC/DC discography"]
    assert calls[1][0].endswith("/page/summary/AC%2FDC")


def test_wikipedia_no_results(fake_http):
    fake_http({"w/api.php": FakeResponse(200, {"query": {"search": []}})})
    assert "No Wikipedia article" in run_tool("search_wikipedia", '{"query": "qwpoeiruty"}')["error"]


def test_wikipedia_bad_json(fake_http):
    fake_http({"w/api.php": FakeResponse(200, None)})
    assert "unexpected response" in run_tool("search_wikipedia", '{"query": "x"}')["error"]


# -------------------------------------------------------------- currency


def test_currency_success(fake_http):
    fake_http({"frankfurter": FakeResponse(200, {"base": "USD", "date": "2026-11-19", "rates": {"INR": 93.3}})})
    out = run_tool("convert_currency", '{"amount": 250, "from_currency": "usd", "to_currency": "INR"}')
    assert out["converted"] == 23325.0 and out["rate_date"] == "2026-11-19"


def test_currency_unsupported_code(fake_http):
    fake_http({"frankfurter": FakeResponse(404, {"message": "not found"})})
    out = run_tool("convert_currency", '{"amount": 5, "from_currency": "USD", "to_currency": "XYZ"}')
    assert "not supported" in out["error"]


def test_currency_validates_input_without_network():
    assert "3-letter" in run_tool("convert_currency", '{"amount": 5, "from_currency": "dollars", "to_currency": "INR"}')["error"]
    assert "negative" in run_tool("convert_currency", '{"amount": -5, "from_currency": "USD", "to_currency": "INR"}')["error"]
    assert convert_currency(10, "EUR", "eur")["converted"] == 10


# --------------------------------------------------------------- run_tool


def test_run_tool_rejects_bad_calls():
    assert "Unknown tool" in run_tool("delete_files", "{}")["error"]
    assert "not valid JSON" in run_tool("calculator", "{expression: 1+1")["error"]
    assert "Missing required argument" in run_tool("calculator", "{}")["error"]
    assert "should be a string" in run_tool("calculator", '{"expression": 5}')["error"]
    assert "should be a number" in run_tool("convert_currency", '{"amount": true, "from_currency": "USD", "to_currency": "INR"}')["error"]
    assert "Unexpected argument" in run_tool("calculator", '{"expression": "1", "x": 2}')["error"]


def test_run_tool_catches_unexpected_crashes(monkeypatch):
    def broken(expression):
        raise RuntimeError("boom")

    monkeypatch.setattr(TOOLS["calculator"], "func", broken)
    assert "failed unexpectedly" in run_tool("calculator", '{"expression": "1"}')["error"]


def test_every_tool_has_a_valid_schema():
    schemas = tool_schemas()
    assert len(schemas) == 4
    for s in schemas:
        fn = s["function"]
        assert s["type"] == "function" and fn["description"]
        assert set(fn["parameters"]["required"]) <= set(fn["parameters"]["properties"])
        for prop in fn["parameters"]["properties"].values():
            assert prop["type"] in ("string", "number") and prop["description"]
