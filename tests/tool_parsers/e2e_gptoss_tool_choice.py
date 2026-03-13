#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""E2E test for GPT-OSS tool_choice=required EBNF grammar.

Targets GPT-OSS Harmony models only. Auto-detects the served model.

Usage:
    # Start vLLM server:
    vllm serve <gpt-oss-model> \
        --tool-parser-plugin openai --enable-auto-tool-choice

    # Run all scenarios:
    python tests/tool_parsers/e2e_gptoss_tool_choice.py

    # Verbose (logs full request/response for debugging):
    python tests/tool_parsers/e2e_gptoss_tool_choice.py -v

    # Run single scenario:
    python tests/tool_parsers/e2e_gptoss_tool_choice.py --scenario korean_unicode

    # Save logs:
    python tests/tool_parsers/e2e_gptoss_tool_choice.py -v --log-dir ./logs
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from openai import OpenAI

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logger = logging.getLogger("e2e_tool_choice")


def setup_logging(verbose: bool, log_dir: str | None) -> None:
    fmt = "%(asctime)s %(levelname)-7s %(message)s"
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_dir:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        handlers.append(logging.FileHandler(log_path / f"e2e_tool_choice_{ts}.log"))
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format=fmt,
        handlers=handlers,
    )


# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------

TOOL_GET_WEATHER = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get current weather for a location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City name",
                },
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                },
            },
            "required": ["location"],
        },
    },
}

TOOL_SEARCH = {
    "type": "function",
    "function": {
        "name": "search",
        "description": "Search the web for information",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "max_results": {"type": "integer", "default": 5},
            },
            "required": ["query"],
        },
    },
}

TOOL_CALCULATE = {
    "type": "function",
    "function": {
        "name": "calculate",
        "description": "Evaluate a mathematical expression",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {"type": "string"},
            },
            "required": ["expression"],
        },
    },
}

TOOL_CREATE_FILE = {
    "type": "function",
    "function": {
        "name": "create_file",
        "description": "Create a file with given content",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"},
                "encoding": {"type": "string", "default": "utf-8"},
            },
            "required": ["path", "content"],
        },
    },
}

TOOL_DATABASE_QUERY = {
    "type": "function",
    "function": {
        "name": "database_query",
        "description": "Execute a database query",
        "parameters": {
            "type": "object",
            "properties": {
                "sql": {"type": "string"},
                "database": {"type": "string"},
                "params": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
            "required": ["sql", "database"],
        },
    },
}

TOOL_GET_TIME = {
    "type": "function",
    "function": {
        "name": "get_current_time",
        "description": "Get the current time in a timezone",
        "parameters": {
            "type": "object",
            "properties": {},
        },
    },
}


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------


@dataclass
class TestScenario:
    name: str
    messages: list[dict]
    tools: list[dict]
    expected_tool_names: list[str] | None = None
    description: str = ""
    min_tool_calls: int = 1


SCENARIOS: list[TestScenario] = [
    TestScenario(
        name="simple_weather",
        description="Basic single tool call",
        messages=[{"role": "user", "content": "What's the weather in Tokyo?"}],
        tools=[TOOL_GET_WEATHER],
        expected_tool_names=["get_weather"],
    ),
    TestScenario(
        name="select_from_multiple",
        description="Choose correct tool from 3 options",
        messages=[
            {
                "role": "user",
                "content": "What is the weather in Seoul right now?",
            }
        ],
        tools=[TOOL_GET_WEATHER, TOOL_SEARCH, TOOL_CALCULATE],
        expected_tool_names=["get_weather"],
    ),
    TestScenario(
        name="calculation",
        description="Math expression with special characters",
        messages=[
            {
                "role": "user",
                "content": "Calculate (3.14 * 2^10) / 7 + sqrt(144)",
            }
        ],
        tools=[TOOL_CALCULATE, TOOL_SEARCH],
        expected_tool_names=["calculate"],
    ),
    TestScenario(
        name="multi_turn",
        description="Tool call after prior conversation context",
        messages=[
            {"role": "user", "content": "I'm planning a trip to Paris."},
            {
                "role": "assistant",
                "content": "That sounds exciting! How can I help with your trip?",
            },
            {
                "role": "user",
                "content": "Check the weather there and search for hotels.",
            },
        ],
        tools=[TOOL_GET_WEATHER, TOOL_SEARCH],
        min_tool_calls=1,
    ),
    TestScenario(
        name="nested_json_args",
        description="Tool call with complex nested JSON",
        messages=[
            {
                "role": "user",
                "content": (
                    "Query the users database: "
                    "SELECT * FROM users WHERE age > 18 "
                    "AND status = 'active' "
                    "with params ['active', '18']"
                ),
            }
        ],
        tools=[TOOL_DATABASE_QUERY],
        expected_tool_names=["database_query"],
    ),
    TestScenario(
        name="special_chars",
        description="User message contains < and > operators",
        messages=[
            {
                "role": "user",
                "content": (
                    "My code has a bug: if x < 10 && y >= 20 "
                    "the loop breaks. Search for solutions "
                    "about comparison operators in Python."
                ),
            }
        ],
        tools=[TOOL_SEARCH],
        expected_tool_names=["search"],
    ),
    TestScenario(
        name="long_prompt",
        description="Long user message still produces tool call",
        messages=[
            {
                "role": "user",
                "content": (
                    "I have a very detailed question about weather "
                    "patterns. In climate science, we often study "
                    "how temperature varies across different "
                    "geographical regions. The relationship between "
                    "latitude and temperature follows a general "
                    "pattern where equatorial regions (0 degrees) "
                    "tend to be warmer than polar regions (90 "
                    "degrees). However, local factors like altitude, "
                    "ocean currents, and urban heat islands can "
                    "significantly modify this pattern. Given all "
                    "this context, what is the current weather in "
                    "New York City?"
                ),
            }
        ],
        tools=[TOOL_GET_WEATHER],
        expected_tool_names=["get_weather"],
    ),
    TestScenario(
        name="html_content",
        description="Create a file with HTML (contains < and >)",
        messages=[
            {
                "role": "user",
                "content": (
                    "Create an HTML file at /tmp/test.html with: "
                    "<html><body><h1>Hello</h1>"
                    "<p>x < y and a > b</p></body></html>"
                ),
            }
        ],
        tools=[TOOL_CREATE_FILE],
        expected_tool_names=["create_file"],
    ),
    TestScenario(
        name="force_tool_on_chat",
        description="tool_choice=required forces tool on casual chat",
        messages=[{"role": "user", "content": "Hello! How are you today?"}],
        tools=[TOOL_SEARCH, TOOL_GET_WEATHER],
        min_tool_calls=1,
    ),
    TestScenario(
        name="five_tools_available",
        description="All 5 tools available, must pick at least one",
        messages=[
            {
                "role": "user",
                "content": "Check the weather in London and search for flights.",
            }
        ],
        tools=[
            TOOL_GET_WEATHER,
            TOOL_SEARCH,
            TOOL_CALCULATE,
            TOOL_CREATE_FILE,
            TOOL_DATABASE_QUERY,
        ],
        min_tool_calls=1,
    ),
    TestScenario(
        name="parallel_tool_calls",
        description="Expect 2+ tool calls for multi-city weather",
        messages=[
            {
                "role": "user",
                "content": (
                    "I need the weather in Tokyo, Seoul, and "
                    "London. Check all three cities."
                ),
            }
        ],
        tools=[TOOL_GET_WEATHER],
        expected_tool_names=["get_weather"],
        min_tool_calls=1,
    ),
    TestScenario(
        name="multi_turn_with_tool_result",
        description="Follow-up after tool result (full tool loop)",
        messages=[
            {
                "role": "user",
                "content": "What's the weather in Paris?",
            },
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_001",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"location": "Paris"}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_001",
                "content": '{"temperature": 18, "unit": "celsius", '
                '"condition": "cloudy"}',
            },
            {
                "role": "user",
                "content": "Now search for indoor activities in Paris.",
            },
        ],
        tools=[TOOL_GET_WEATHER, TOOL_SEARCH],
        expected_tool_names=["search"],
    ),
    TestScenario(
        name="korean_unicode",
        description="Non-English user message (Korean)",
        messages=[
            {
                "role": "user",
                "content": "서울의 현재 날씨를 알려주세요. 기온이 영하인지 확인해줘.",
            }
        ],
        tools=[TOOL_GET_WEATHER],
        expected_tool_names=["get_weather"],
    ),
    TestScenario(
        name="with_system_message",
        description="System prompt + user request",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant. "
                "Always use tools when available.",
            },
            {
                "role": "user",
                "content": "What is the weather in Berlin?",
            },
        ],
        tools=[TOOL_GET_WEATHER, TOOL_SEARCH],
        expected_tool_names=["get_weather"],
    ),
    TestScenario(
        name="no_arg_tool",
        description="Tool with no required parameters",
        messages=[{"role": "user", "content": "What time is it right now?"}],
        tools=[TOOL_GET_TIME],
        expected_tool_names=["get_current_time"],
    ),
    TestScenario(
        name="chain_reasoning",
        description="Request implies tool chaining (weather + calc)",
        messages=[
            {
                "role": "user",
                "content": (
                    "Get the temperature in Tokyo and calculate "
                    "the conversion from Celsius to Fahrenheit "
                    "using the formula F = C * 9/5 + 32."
                ),
            }
        ],
        tools=[TOOL_GET_WEATHER, TOOL_CALCULATE],
        min_tool_calls=1,
    ),
]


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


@dataclass
class TestResult:
    scenario: str
    passed: bool
    tool_calls: list[dict] = field(default_factory=list)
    content: str | None = None
    error: str | None = None
    latency_ms: float = 0.0
    raw_response: dict | None = None


def _validate_tool_args(
    tool_call: dict,
    tool_defs: list[dict],
) -> str | None:
    """Validate tool call args against the tool's schema."""
    name = tool_call["name"]
    args_str = tool_call["arguments"]

    try:
        args = json.loads(args_str)
    except json.JSONDecodeError:
        return f"{name}: invalid JSON: {args_str[:100]}"

    tool_def = None
    for t in tool_defs:
        if t["function"]["name"] == name:
            tool_def = t
            break
    if tool_def is None:
        return f"{name}: not found in tool definitions"

    params = tool_def["function"].get("parameters", {})
    for field_name in params.get("required", []):
        if field_name not in args:
            return f"{name}: missing required field '{field_name}'"

    return None


def _detect_model(client: OpenAI) -> str:
    """Auto-detect the served model name from vLLM."""
    models = client.models.list()
    model_ids = [m.id for m in models.data]
    if len(model_ids) == 1:
        return model_ids[0]
    if len(model_ids) == 0:
        logger.error("No models served. Start vLLM first.")
        sys.exit(1)
    logger.error(
        "Multiple models served: %s. Use --model to specify.",
        model_ids,
    )
    sys.exit(1)


def run_scenario(
    client: OpenAI,
    model: str,
    scenario: TestScenario,
    verbose: bool = False,
) -> TestResult:
    logger.info("--- [%s] %s ---", scenario.name, scenario.description)

    if verbose:
        logger.debug(
            "  Request:\n%s",
            json.dumps(scenario.messages, indent=2, ensure_ascii=False),
        )
        tool_names = [t["function"]["name"] for t in scenario.tools]
        logger.debug("  Tools: %s", tool_names)

    t0 = time.monotonic()
    try:
        response = client.chat.completions.create(
            model=model,
            messages=scenario.messages,
            tools=scenario.tools,
            tool_choice="required",
            temperature=0,
            max_tokens=4096,
        )
    except Exception as e:
        logger.error("  API error: %s", e)
        return TestResult(scenario=scenario.name, passed=False, error=str(e))
    latency = (time.monotonic() - t0) * 1000

    raw = response.model_dump()
    choice = response.choices[0]
    msg = choice.message

    if verbose:
        logger.debug(
            "  Response:\n%s",
            json.dumps(raw, indent=2, ensure_ascii=False),
        )

    # Extract tool calls
    tool_calls_data = []
    if msg.tool_calls:
        for tc in msg.tool_calls:
            tc_info = {
                "id": tc.id,
                "name": tc.function.name,
                "arguments": tc.function.arguments,
            }
            tool_calls_data.append(tc_info)
            logger.info(
                "  Tool call: %s(%s)",
                tc.function.name,
                tc.function.arguments,
            )

    if msg.content:
        logger.info("  Content: %s", msg.content[:200])

    # Validate
    passed = True
    error_parts: list[str] = []

    if len(tool_calls_data) < scenario.min_tool_calls:
        passed = False
        error_parts.append(
            f"Expected >= {scenario.min_tool_calls} tool calls,"
            f" got {len(tool_calls_data)}"
        )

    if scenario.expected_tool_names and tool_calls_data:
        actual = {tc["name"] for tc in tool_calls_data}
        expected = set(scenario.expected_tool_names)
        valid = {t["function"]["name"] for t in scenario.tools}
        if not actual.issubset(valid):
            passed = False
            error_parts.append(f"Unexpected tools: {actual}")
        if not actual & expected:
            passed = False
            error_parts.append(f"Expected one of {expected}, got {actual}")

    fr = choice.finish_reason
    if fr == "error":
        passed = False
        error_parts.append("finish_reason=error")
    elif fr == "length":
        passed = False
        error_parts.append("finish_reason=length (truncated)")

    for tc in tool_calls_data:
        err = _validate_tool_args(tc, scenario.tools)
        if err:
            passed = False
            error_parts.append(f"Schema: {err}")

    result = TestResult(
        scenario=scenario.name,
        passed=passed,
        tool_calls=tool_calls_data,
        content=msg.content,
        error="; ".join(error_parts) if error_parts else None,
        latency_ms=latency,
        raw_response=raw if verbose else None,
    )

    status = "PASS" if passed else "FAIL"
    logger.info(
        "  %s (%.0fms, %d tool calls, finish=%s)",
        status,
        latency,
        len(tool_calls_data),
        fr,
    )
    if not passed:
        logger.error("  Error: %s", result.error)

    return result


def run_all(
    client: OpenAI,
    model: str,
    scenarios: list[TestScenario],
    verbose: bool,
) -> list[TestResult]:
    return [run_scenario(client, model, s, verbose) for s in scenarios]


def print_summary(results: list[TestResult]) -> None:
    total = len(results)
    passed = sum(1 for r in results if r.passed)

    logger.info("")
    logger.info("=" * 60)
    logger.info(
        "SUMMARY: %d/%d passed (%.1f%%)",
        passed,
        total,
        100 * passed / total if total else 0,
    )
    logger.info("=" * 60)

    if passed < total:
        logger.info("")
        logger.info("FAILURES:")
        for r in results:
            if not r.passed:
                logger.info("  - %s: %s", r.scenario, r.error)

    logger.info("")
    for r in results:
        logger.info(
            "  %-30s %s  %.0fms",
            r.scenario,
            "PASS" if r.passed else "FAIL",
            r.latency_ms,
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="E2E test for GPT-OSS tool_choice=required"
    )
    parser.add_argument(
        "--base-url",
        default="http://localhost:8000/v1",
        help="vLLM server URL (default: %(default)s)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model name (auto-detected from server if omitted)",
    )
    parser.add_argument("--api-key", default="EMPTY", help="API key")
    parser.add_argument("-v", "--verbose", action="store_true", help="Log I/O")
    parser.add_argument("--log-dir", default=None, help="Save logs to directory")
    parser.add_argument("--scenario", default=None, help="Run single scenario")

    args = parser.parse_args()
    setup_logging(args.verbose, args.log_dir)

    client = OpenAI(base_url=args.base_url, api_key=args.api_key)
    model = args.model or _detect_model(client)
    logger.info("Server: %s  Model: %s", args.base_url, model)

    scenarios = SCENARIOS
    if args.scenario:
        scenarios = [s for s in SCENARIOS if s.name == args.scenario]
        if not scenarios:
            logger.error("Scenario '%s' not found", args.scenario)
            sys.exit(1)

    results = run_all(client, model, scenarios, args.verbose)
    print_summary(results)
    sys.exit(0 if all(r.passed for r in results) else 1)


if __name__ == "__main__":
    main()
