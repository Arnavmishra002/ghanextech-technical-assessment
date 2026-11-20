"""The tools the agent can call.

Each tool is a plain Python function plus a JSON schema that tells the LLM
what the tool does and what arguments it takes. run_tool() is the only entry
point the agent uses: it never raises, it always returns a dict, and on
failure that dict has an "error" key the model can explain to the user.
"""

import ast
import json
import math
import operator
from dataclasses import dataclass
from typing import Callable
from urllib.parse import quote

import requests

TIMEOUT_SECONDS = 10
HEADERS = {"User-Agent": "GhanexAssessmentAgent/1.0 (educational demo)"}


class ToolError(Exception):
    """A tool failure with a message that is fine to show to the user."""

    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status


def http_get_json(url: str, params: dict) -> dict:
    try:
        response = requests.get(url, params=params, headers=HEADERS, timeout=TIMEOUT_SECONDS)
    except requests.Timeout as exc:
        raise ToolError("The service took too long to answer. Please try again.") from exc
    except requests.RequestException as exc:
        raise ToolError("Network problem while contacting the service.") from exc
    if response.status_code == 429:
        raise ToolError("The service is getting too many requests. Please wait a moment.", 429)
    if response.status_code >= 400:
        raise ToolError(f"The service returned an error (HTTP {response.status_code}).", response.status_code)
    try:
        return response.json()
    except ValueError as exc:
        raise ToolError("The service sent back an unexpected response.") from exc


# ---------------------------------------------------------------- calculator

_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_FUNCTIONS = {
    "sqrt": math.sqrt,
    "abs": abs,
    "round": round,
    "floor": math.floor,
    "ceil": math.ceil,
    "log": math.log,
    "log10": math.log10,
    "exp": math.exp,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
}
_CONSTANTS = {"pi": math.pi, "e": math.e}
MAX_EXPONENT = 1000
MAX_EXPRESSION_LENGTH = 200


def _evaluate(node):
    if isinstance(node, ast.Expression):
        return _evaluate(node.body)
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return node.value
    if isinstance(node, ast.Name) and node.id in _CONSTANTS:
        return _CONSTANTS[node.id]
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_evaluate(node.operand))
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        left, right = _evaluate(node.left), _evaluate(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > MAX_EXPONENT:
            raise ToolError(f"Exponent too large (limit is {MAX_EXPONENT}).")
        return _BIN_OPS[type(node.op)](left, right)
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in _FUNCTIONS
        and not node.keywords
    ):
        return _FUNCTIONS[node.func.id](*[_evaluate(a) for a in node.args])
    raise ToolError("Only numbers, + - * / // % **, brackets and basic math functions are allowed.")


def calculator(expression: str) -> dict:
    """Evaluate a maths expression safely (no eval, only whitelisted syntax)."""
    expression = expression.strip().replace("^", "**").replace("×", "*").replace("÷", "/")
    if not expression:
        raise ToolError("The expression is empty.")
    if len(expression) > MAX_EXPRESSION_LENGTH:
        raise ToolError("The expression is too long.")
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ToolError(f"'{expression}' is not a valid expression.") from exc
    try:
        result = _evaluate(tree)
    except ZeroDivisionError as exc:
        raise ToolError("Division by zero.") from exc
    except OverflowError as exc:
        raise ToolError("The result is too large.") from exc
    except (ValueError, TypeError) as exc:
        raise ToolError(f"Math error: {exc}.") from exc
    if isinstance(result, float) and result.is_integer() and abs(result) < 1e15:
        result = int(result)
    return {"expression": expression, "result": result}


# ------------------------------------------------------------------- weather

WEATHER_CODES = {
    0: "clear sky", 1: "mainly clear", 2: "partly cloudy", 3: "overcast",
    45: "fog", 48: "freezing fog",
    51: "light drizzle", 53: "drizzle", 55: "heavy drizzle",
    56: "freezing drizzle", 57: "heavy freezing drizzle",
    61: "light rain", 63: "rain", 65: "heavy rain",
    66: "freezing rain", 67: "heavy freezing rain",
    71: "light snow", 73: "snow", 75: "heavy snow", 77: "snow grains",
    80: "light rain showers", 81: "rain showers", 82: "violent rain showers",
    85: "snow showers", 86: "heavy snow showers",
    95: "thunderstorm", 96: "thunderstorm with hail", 99: "thunderstorm with heavy hail",
}


def get_weather(location: str) -> dict:
    """Current weather and today's forecast from Open-Meteo (free, no key)."""
    location = location.strip()
    if not location:
        raise ToolError("Please give a place name.")
    geo = http_get_json(
        "https://geocoding-api.open-meteo.com/v1/search",
        {"name": location, "count": 1, "language": "en", "format": "json"},
    )
    if not geo.get("results"):
        raise ToolError(f"Could not find a place called '{location}'. Try a nearby city name.")
    place = geo["results"][0]
    data = http_get_json(
        "https://api.open-meteo.com/v1/forecast",
        {
            "latitude": place["latitude"],
            "longitude": place["longitude"],
            "current": "temperature_2m,apparent_temperature,relative_humidity_2m,wind_speed_10m,weather_code",
            "daily": "temperature_2m_max,temperature_2m_min,precipitation_probability_max",
            "forecast_days": 1,
            "timezone": "auto",
        },
    )
    current, daily = data["current"], data["daily"]
    return {
        "place": ", ".join(p for p in (place.get("name"), place.get("admin1"), place.get("country")) if p),
        "local_time": current["time"],
        "condition": WEATHER_CODES.get(current["weather_code"], "unknown"),
        "temperature_c": current["temperature_2m"],
        "feels_like_c": current["apparent_temperature"],
        "humidity_percent": current["relative_humidity_2m"],
        "wind_kmh": current["wind_speed_10m"],
        "today_max_c": daily["temperature_2m_max"][0],
        "today_min_c": daily["temperature_2m_min"][0],
        "rain_chance_percent": daily["precipitation_probability_max"][0],
        "source": "Open-Meteo",
    }


# ----------------------------------------------------------------- wikipedia

def search_wikipedia(query: str) -> dict:
    """Find the best matching Wikipedia article and return its summary."""
    query = query.strip()
    if not query:
        raise ToolError("Please give something to search for.")
    found = http_get_json(
        "https://en.wikipedia.org/w/api.php",
        {"action": "query", "list": "search", "srsearch": query, "srlimit": 3, "format": "json"},
    )
    hits = found.get("query", {}).get("search", [])
    if not hits:
        raise ToolError(f"No Wikipedia article found for '{query}'.")
    title = hits[0]["title"]
    summary = http_get_json(
        # safe="" also encodes "/" so titles like "AC/DC" work.
        "https://en.wikipedia.org/api/rest_v1/page/summary/" + quote(title.replace(" ", "_"), safe=""),
        {},
    )
    return {
        "title": summary.get("title", title),
        "summary": (summary.get("extract") or "")[:1500],
        "url": summary.get("content_urls", {}).get("desktop", {}).get("page", ""),
        "is_disambiguation": summary.get("type") == "disambiguation",
        "other_matches": [h["title"] for h in hits[1:]],
    }


# ------------------------------------------------------------------ currency

def convert_currency(amount: float, from_currency: str, to_currency: str) -> dict:
    """Convert money using European Central Bank reference rates (Frankfurter API)."""
    source, target = from_currency.strip().upper(), to_currency.strip().upper()
    for code in (source, target):
        if len(code) != 3 or not code.isalpha():
            raise ToolError(f"'{code}' is not a 3-letter currency code like USD or INR.")
    if amount < 0:
        raise ToolError("The amount cannot be negative.")
    if source == target:
        return {"amount": amount, "from": source, "to": target, "rate": 1.0, "converted": amount}
    try:
        data = http_get_json(
            "https://api.frankfurter.dev/v1/latest", {"base": source, "symbols": target}
        )
    except ToolError as exc:
        if exc.status in (404, 422):
            raise ToolError(
                f"{source} or {target} is not supported. The rates cover about 30 major currencies."
            ) from exc
        raise
    rate = data.get("rates", {}).get(target)
    if rate is None:
        raise ToolError(f"No exchange rate available for {source} to {target}.")
    return {
        "amount": amount,
        "from": source,
        "to": target,
        "rate": rate,
        "converted": round(amount * rate, 2),
        "rate_date": data.get("date"),
        "source": "European Central Bank reference rates (updated once per working day)",
    }


# ------------------------------------------------------------------ registry

@dataclass
class Tool:
    name: str
    description: str
    parameters: dict
    func: Callable[..., dict]

    def schema(self) -> dict:
        return {
            "type": "function",
            "function": {"name": self.name, "description": self.description, "parameters": self.parameters},
        }


TOOLS: dict[str, Tool] = {
    t.name: t
    for t in [
        Tool(
            "calculator",
            "Evaluate a maths expression exactly. Use for any arithmetic, percentages, powers, "
            "square roots or trigonometry instead of calculating in your head.",
            {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Python-style maths, e.g. '(1250 * 0.18) + 40' or 'sqrt(2) ** 3'. "
                        "Allowed functions: sqrt, abs, round, floor, ceil, log, log10, exp, sin, cos, tan; constants pi, e.",
                    }
                },
                "required": ["expression"],
            },
            calculator,
        ),
        Tool(
            "get_weather",
            "Get the current weather and today's high, low and rain chance for a city or place.",
            {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City or place name, e.g. 'Pune' or 'Paris, France'."}
                },
                "required": ["location"],
            },
            get_weather,
        ),
        Tool(
            "search_wikipedia",
            "Look up a topic, person, place or event on Wikipedia and get a short summary. "
            "Use for factual questions you are not sure about or that may have changed recently.",
            {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "What to search for, e.g. 'Chandrayaan-3'."}
                },
                "required": ["query"],
            },
            search_wikipedia,
        ),
        Tool(
            "convert_currency",
            "Convert an amount of money from one currency to another using the latest daily exchange rate.",
            {
                "type": "object",
                "properties": {
                    "amount": {"type": "number", "description": "Amount of money to convert."},
                    "from_currency": {"type": "string", "description": "3-letter code of the source currency, e.g. USD."},
                    "to_currency": {"type": "string", "description": "3-letter code of the target currency, e.g. INR."},
                },
                "required": ["amount", "from_currency", "to_currency"],
            },
            convert_currency,
        ),
    ]
}

_JSON_TYPES = {"string": str, "number": (int, float)}


def tool_schemas() -> list[dict]:
    return [t.schema() for t in TOOLS.values()]


def run_tool(name: str, raw_arguments: str) -> dict:
    """Run a tool call coming from the model. Never raises."""
    tool = TOOLS.get(name)
    if tool is None:
        return {"error": f"Unknown tool '{name}'."}
    try:
        args = json.loads(raw_arguments or "{}")
    except json.JSONDecodeError:
        return {"error": "The tool arguments were not valid JSON."}
    if not isinstance(args, dict):
        return {"error": "The tool arguments must be a JSON object."}

    props = tool.parameters["properties"]
    for key in tool.parameters["required"]:
        if key not in args:
            return {"error": f"Missing required argument '{key}'."}
    for key, value in args.items():
        if key not in props:
            return {"error": f"Unexpected argument '{key}'."}
        expected = _JSON_TYPES[props[key]["type"]]
        if isinstance(value, bool) or not isinstance(value, expected):
            return {"error": f"Argument '{key}' should be a {props[key]['type']}."}

    try:
        return tool.func(**args)
    except ToolError as exc:
        return {"error": str(exc)}
    except Exception as exc:  # last line of defence: the chat must not crash
        return {"error": f"The {name} tool failed unexpectedly ({type(exc).__name__})."}
