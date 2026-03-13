#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""E2E test script for GPT-OSS tool_choice=required with EBNF grammar.

Usage:
    # Start vLLM server first:
    vllm serve <model> --tool-parser-plugin openai --enable-auto-tool-choice

    # Run tests:
    python tests/tool_parsers/e2e_gptoss_tool_choice.py \
        --base-url http://localhost:8000/v1 \
        --model <model_name> \
        --iterations 3 \
        --verbose

    # Compare baseline (unpatched) vs patched:
    python tests/tool_parsers/e2e_gptoss_tool_choice.py \
        --base-url http://localhost:8000/v1 \
        --model <model_name> \
        --log-dir ./logs
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
# Logging setup
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


# ---------------------------------------------------------------------------
# Test scenarios
# ---------------------------------------------------------------------------


@dataclass
class TestScenario:
    name: str
    messages: list[dict]
    tools: list[dict]
    expected_tool_names: list[str] | None = None  # None = any tool is OK
    description: str = ""
    min_tool_calls: int = 1


SCENARIOS: list[TestScenario] = [
    # 1. Simple single tool call
    TestScenario(
        name="simple_weather",
        description="Basic single tool call",
        messages=[{"role": "user", "content": "What's the weather in Tokyo?"}],
        tools=[TOOL_GET_WEATHER],
        expected_tool_names=["get_weather"],
    ),
    # 2. Single tool from multiple available
    TestScenario(
        name="select_from_multiple",
        description="Choose correct tool from 3 options",
        messages=[
            {"role": "user", "content": "What is the weather in Seoul right now?"}
        ],
        tools=[TOOL_GET_WEATHER, TOOL_SEARCH, TOOL_CALCULATE],
        expected_tool_names=["get_weather"],
    ),
    # 3. Calculation tool
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
    # 4. Multi-turn conversation
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
    # 5. Nested JSON arguments
    TestScenario(
        name="nested_json_args",
        description="Tool call with complex nested JSON",
        messages=[
            {
                "role": "user",
                "content": (
                    "Query the users database: "
                    "SELECT * FROM users WHERE age > 18 AND status = 'active' "
                    "with params ['active', '18']"
                ),
            }
        ],
        tools=[TOOL_DATABASE_QUERY],
        expected_tool_names=["database_query"],
    ),
    # 6. Content with special characters (< > = comparisons)
    TestScenario(
        name="special_chars",
        description="User message contains < and > operators",
        messages=[
            {
                "role": "user",
                "content": (
                    "My code has a bug: if x < 10 && y >= 20 the loop breaks. "
                    "Search for solutions about comparison operators in Python."
                ),
            }
        ],
        tools=[TOOL_SEARCH],
        expected_tool_names=["search"],
    ),
    # 7. Long content / complex prompt
    TestScenario(
        name="long_prompt",
        description="Long user message that should still produce tool call",
        messages=[
            {
                "role": "user",
                "content": (
                    "I have a very detailed question about weather patterns. "
                    "In climate science, we often study how temperature varies "
                    "across different geographical regions. The relationship "
                    "between latitude and temperature follows a general pattern "
                    "where equatorial regions (0 degrees) tend to be warmer "
                    "than polar regions (90 degrees). However, local factors "
                    "like altitude, ocean currents, and urban heat islands can "
                    "significantly modify this pattern. Given all this context, "
                    "what is the current weather in New York City?"
                ),
            }
        ],
        tools=[TOOL_GET_WEATHER],
        expected_tool_names=["get_weather"],
    ),
    # 8. File creation with HTML content
    TestScenario(
        name="html_content",
        description="Create a file with HTML (contains < and >)",
        messages=[
            {
                "role": "user",
                "content": (
                    "Create an HTML file at /tmp/test.html with content: "
                    "<html><body><h1>Hello</h1><p>x < y and a > b</p></body></html>"
                ),
            }
        ],
        tools=[TOOL_CREATE_FILE],
        expected_tool_names=["create_file"],
    ),
    # 9. Must call tool even for conversational input
    TestScenario(
        name="force_tool_on_chat",
        description="tool_choice=required forces tool call even for casual chat",
        messages=[{"role": "user", "content": "Hello! How are you today?"}],
        tools=[TOOL_SEARCH, TOOL_GET_WEATHER],
        min_tool_calls=1,
    ),
    # 10. Multi-tool scenario with all 5 tools
    TestScenario(
        name="five_tools_available",
        description="All 5 tools available, model must pick at least one",
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
]


# ---------------------------------------------------------------------------
# Test runner
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


def run_scenario(
    client: OpenAI,
    model: str,
    scenario: TestScenario,
    verbose: bool = False,
) -> TestResult:
    logger.info("--- [%s] %s ---", scenario.name, scenario.description)

    t0 = time.monotonic()
    try:
        response = client.chat.completions.create(
            model=model,
            messages=scenario.messages,
            tools=scenario.tools,
            tool_choice="required",
        )
    except Exception as e:
        logger.error("  API error: %s", e)
        return TestResult(scenario=scenario.name, passed=False, error=str(e))
    latency = (time.monotonic() - t0) * 1000

    raw = response.model_dump()
    choice = response.choices[0]
    msg = choice.message

    # Log raw response
    if verbose:
        logger.debug(
            "  Raw response:\n%s", json.dumps(raw, indent=2, ensure_ascii=False)
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
            logger.info("  Tool call: %s(%s)", tc.function.name, tc.function.arguments)

            # Validate JSON arguments
            try:
                json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                logger.warning(
                    "  WARNING: Invalid JSON in arguments: %s",
                    tc.function.arguments,
                )

    content = msg.content
    if content:
        logger.info("  Content: %s", content[:200])

    # Validate
    passed = True
    error_parts = []

    if len(tool_calls_data) < scenario.min_tool_calls:
        passed = False
        got = len(tool_calls_data)
        error_parts.append(
            f"Expected >= {scenario.min_tool_calls} tool calls, got {got}"
        )

    if scenario.expected_tool_names and tool_calls_data:
        actual_names = {tc["name"] for tc in tool_calls_data}
        expected = set(scenario.expected_tool_names)
        valid = {t["function"]["name"] for t in scenario.tools}
        if not actual_names.issubset(valid):
            passed = False
            error_parts.append(f"Got unexpected tool names: {actual_names}")
        # Check at least one expected tool was called
        if expected and not actual_names & expected:
            passed = False
            error_parts.append(f"Expected one of {expected}, got {actual_names}")

    if choice.finish_reason == "error":
        passed = False
        error_parts.append("finish_reason=error")

    result = TestResult(
        scenario=scenario.name,
        passed=passed,
        tool_calls=tool_calls_data,
        content=content,
        error="; ".join(error_parts) if error_parts else None,
        latency_ms=latency,
        raw_response=raw if verbose else None,
    )

    status = "PASS" if passed else "FAIL"
    logger.info(
        "  Result: %s (%.0fms, %d tool calls)",
        status,
        latency,
        len(tool_calls_data),
    )
    if not passed:
        logger.error("  Error: %s", result.error)

    return result


def run_all(
    client: OpenAI,
    model: str,
    iterations: int,
    verbose: bool,
) -> list[TestResult]:
    all_results: list[TestResult] = []

    for i in range(iterations):
        logger.info("========== Iteration %d/%d ==========", i + 1, iterations)
        for scenario in SCENARIOS:
            result = run_scenario(client, model, scenario, verbose)
            all_results.append(result)

    return all_results


def print_summary(results: list[TestResult], iterations: int) -> None:
    total = len(results)
    passed = sum(1 for r in results if r.passed)
    failed = total - passed

    logger.info("")
    logger.info("=" * 60)
    logger.info("SUMMARY: %d/%d passed (%.1f%%)", passed, total, 100 * passed / total)
    logger.info("Iterations: %d, Scenarios: %d", iterations, len(SCENARIOS))
    logger.info("=" * 60)

    if failed > 0:
        logger.info("")
        logger.info("FAILURES:")
        for r in results:
            if not r.passed:
                logger.info("  - %s: %s", r.scenario, r.error)

    # Per-scenario breakdown
    logger.info("")
    logger.info("Per-scenario results:")
    scenario_names = [s.name for s in SCENARIOS]
    for name in scenario_names:
        runs = [r for r in results if r.scenario == name]
        p = sum(1 for r in runs if r.passed)
        avg_ms = sum(r.latency_ms for r in runs) / len(runs) if runs else 0
        logger.info(
            "  %-25s %d/%d passed  avg %.0fms",
            name,
            p,
            len(runs),
            avg_ms,
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
        help="vLLM server base URL",
    )
    parser.add_argument("--model", required=True, help="Model name")
    parser.add_argument("--api-key", default="EMPTY", help="API key")
    parser.add_argument(
        "--iterations", type=int, default=1, help="Number of iterations"
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    parser.add_argument("--log-dir", default=None, help="Directory to save log files")
    parser.add_argument(
        "--scenario",
        default=None,
        help="Run only this scenario (by name)",
    )

    args = parser.parse_args()
    setup_logging(args.verbose, args.log_dir)

    logger.info("Connecting to %s (model=%s)", args.base_url, args.model)

    client = OpenAI(base_url=args.base_url, api_key=args.api_key)

    # Filter scenarios if specified
    if args.scenario:
        global SCENARIOS
        SCENARIOS = [s for s in SCENARIOS if s.name == args.scenario]
        if not SCENARIOS:
            logger.error("Scenario '%s' not found", args.scenario)
            sys.exit(1)

    results = run_all(client, args.model, args.iterations, args.verbose)
    print_summary(results, args.iterations)

    # Exit code
    if all(r.passed for r in results):
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
