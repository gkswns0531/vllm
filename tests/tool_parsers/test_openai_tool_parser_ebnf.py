# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for OpenAIToolParser EBNF grammar generation and validation.

Tests the _build_tool_required_grammar() method and validates the grammar
against xgrammar to ensure it correctly:
- Blocks the final channel
- Allows multi-round analysis (reasoning)
- Allows commentary preambles
- Allows tool calls (commentary to=functions.X)
- Requires at least one tool call before termination
- Handles multiple tool calls
- Allows '<' in content (comparisons, HTML, etc.)
"""

import pytest

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
    ChatCompletionToolsParam,
    FunctionDefinition,
)
from vllm.sampling_params import StructuredOutputsParams
from vllm.tokenizers import get_tokenizer
from vllm.tool_parsers.openai_tool_parser import OpenAIToolParser

MODEL = "gpt2"


@pytest.fixture(scope="module")
def tokenizer():
    return get_tokenizer(MODEL)


@pytest.fixture
def parser(tokenizer):
    return OpenAIToolParser(tokenizer)


# ---------------------------------------------------------------------------
# Grammar generation tests
# ---------------------------------------------------------------------------


def test_build_grammar_single_tool(parser: OpenAIToolParser) -> None:
    grammar = parser._build_tool_required_grammar(["get_weather"])
    assert '"functions.get_weather"' in grammar
    assert "root ::=" in grammar
    assert "tool_block" in grammar
    assert "non_tool_block" in grammar
    assert "content" in grammar


def test_build_grammar_multiple_tools(parser: OpenAIToolParser) -> None:
    tools = ["get_weather", "search", "calculate"]
    grammar = parser._build_tool_required_grammar(tools)
    for name in tools:
        assert f'"functions.{name}"' in grammar
    # Check alternatives are separated by |
    assert (
        '"functions.get_weather" | "functions.search" | "functions.calculate"'
        in grammar
    )


def test_build_grammar_has_content_rule(parser: OpenAIToolParser) -> None:
    grammar = parser._build_tool_required_grammar(["f"])
    # Content rule allows '<' when not followed by '|'
    assert '([^<] | "<" [^|])*' in grammar


def test_build_grammar_no_final_channel(parser: OpenAIToolParser) -> None:
    grammar = parser._build_tool_required_grammar(["f"])
    assert '"final"' not in grammar
    assert "<|return|>" not in grammar


# ---------------------------------------------------------------------------
# adjust_request tests
# ---------------------------------------------------------------------------


def _make_tools(*names: str) -> list[ChatCompletionToolsParam]:
    return [
        ChatCompletionToolsParam(
            type="function",
            function=FunctionDefinition(
                name=name,
                description=f"Tool {name}",
                parameters={
                    "type": "object",
                    "properties": {"q": {"type": "string"}},
                },
            ),
        )
        for name in names
    ]


def test_adjust_request_required(parser: OpenAIToolParser) -> None:
    request = ChatCompletionRequest(
        model="test",
        messages=[{"role": "user", "content": "hi"}],
        tools=_make_tools("get_weather", "search"),
        tool_choice="required",
    )
    result = parser.adjust_request(request)
    assert result.structured_outputs is not None
    assert isinstance(result.structured_outputs, StructuredOutputsParams)
    assert result.structured_outputs.grammar is not None
    assert '"functions.get_weather"' in result.structured_outputs.grammar
    assert '"functions.search"' in result.structured_outputs.grammar
    assert result.response_format is None


def test_adjust_request_auto_unchanged(parser: OpenAIToolParser) -> None:
    request = ChatCompletionRequest(
        model="test",
        messages=[{"role": "user", "content": "hi"}],
        tools=_make_tools("f"),
        tool_choice="auto",
    )
    result = parser.adjust_request(request)
    assert result.structured_outputs is None


def test_adjust_request_no_tools_unchanged(parser: OpenAIToolParser) -> None:
    request = ChatCompletionRequest(
        model="test",
        messages=[{"role": "user", "content": "hi"}],
    )
    result = parser.adjust_request(request)
    assert result.structured_outputs is None


# ---------------------------------------------------------------------------
# xgrammar validation tests (require xgrammar installed)
# ---------------------------------------------------------------------------

xgrammar = pytest.importorskip("xgrammar")


def _make_test_vocab() -> list[str]:
    """Simulated GPT-OSS vocabulary with Harmony special tokens."""
    return [
        "<|end|>",  # 0
        "<|start|>",  # 1
        "<|channel|>",  # 2
        "<|message|>",  # 3
        "<|return|>",  # 4
        "<|call|>",  # 5
        "assistant",  # 6
        "analysis",  # 7
        "commentary",  # 8
        "final",  # 9
        " to=",  # 10
        "functions.",  # 11
        "get_weather",  # 12
        "search",  # 13
        "calculate",  # 14
        "I need",  # 15
        " to check",  # 16
        " the weather",  # 17
        "{",  # 18
        "}",  # 19
        '"',  # 20
        "location",  # 21
        ":",  # 22
        "Tokyo",  # 23
        "<|eos|>",  # 24
        "Let me",  # 25
        " call",  # 26
        " tools",  # 27
        " < ",  # 28  — comparison operator
        "x",  # 29
        "<=",  # 30
        "</div>",  # 31
        " ",  # 32
        "query",  # 33
        "hello",  # 34
        "world",  # 35
        "a",  # 36
        "b",  # 37
        ",",  # 38
        " and",  # 39
    ]


@pytest.fixture(scope="module")
def xgr_compiler():
    vocab = _make_test_vocab()
    tokenizer_info = xgrammar.TokenizerInfo(
        encoded_vocab=vocab,
        vocab_type=xgrammar.VocabType.RAW,
        vocab_size=len(vocab),
        stop_token_ids=[24],
    )
    return xgrammar.GrammarCompiler(tokenizer_info)


def _compile_and_run(
    compiler,
    tool_names: list[str],
    token_ids: list[int],
) -> bool:
    """Compile grammar and test if token sequence is accepted."""
    grammar = OpenAIToolParser._build_tool_required_grammar(tool_names)
    ctx = compiler.compile_grammar(grammar)
    matcher = xgrammar.GrammarMatcher(ctx)
    return all(matcher.accept_token(tid) for tid in token_ids)


def _compile_and_check_blocked(
    compiler,
    tool_names: list[str],
    token_ids: list[int],
    blocked_at: int,
) -> None:
    """Check that a specific token is blocked."""
    grammar = OpenAIToolParser._build_tool_required_grammar(tool_names)
    ctx = compiler.compile_grammar(grammar)
    matcher = xgrammar.GrammarMatcher(ctx)
    for i, tid in enumerate(token_ids):
        result = matcher.accept_token(tid)
        if i == blocked_at:
            assert not result, f"Token {tid} at position {i} should be blocked"
            return
        assert result, f"Token {tid} at position {i} should be accepted"


VOCAB = _make_test_vocab()
V = {s: i for i, s in enumerate(VOCAB)}


class TestXgrammarAcceptance:
    """Test that the grammar accepts valid sequences and blocks invalid ones."""

    def test_direct_tool_call(self, xgr_compiler) -> None:
        """Simplest case: model directly makes a tool call."""
        seq = [
            V["commentary"],
            V[" to="],
            V["functions."],
            V["get_weather"],
            V["<|message|>"],
            V["{"],
            V['"'],
            V["location"],
            V['"'],
            V[":"],
            V['"'],
            V["Tokyo"],
            V['"'],
            V["}"],
            V["<|end|>"],
            V["<|call|>"],
        ]
        assert _compile_and_run(xgr_compiler, ["get_weather"], seq)

    def test_analysis_then_tool_call(self, xgr_compiler) -> None:
        """Single analysis round followed by tool call."""
        seq = [
            # analysis
            V["analysis"],
            V["<|message|>"],
            V["I need"],
            V[" to check"],
            V[" the weather"],
            V["<|end|>"],
            V["<|start|>"],
            V["assistant"],
            V["<|channel|>"],
            # tool call
            V["commentary"],
            V[" to="],
            V["functions."],
            V["get_weather"],
            V["<|message|>"],
            V["{"],
            V["}"],
            V["<|end|>"],
            V["<|call|>"],
        ]
        assert _compile_and_run(xgr_compiler, ["get_weather"], seq)

    def test_multi_round_analysis(self, xgr_compiler) -> None:
        """Multiple analysis rounds before tool call."""
        seq = [
            # analysis round 1
            V["analysis"],
            V["<|message|>"],
            V["I need"],
            V[" to check"],
            V["<|end|>"],
            V["<|start|>"],
            V["assistant"],
            V["<|channel|>"],
            # analysis round 2
            V["analysis"],
            V["<|message|>"],
            V["Let me"],
            V[" call"],
            V[" tools"],
            V["<|end|>"],
            V["<|start|>"],
            V["assistant"],
            V["<|channel|>"],
            # tool call
            V["commentary"],
            V[" to="],
            V["functions."],
            V["search"],
            V["<|message|>"],
            V["{"],
            V["}"],
            V["<|end|>"],
            V["<|call|>"],
        ]
        assert _compile_and_run(xgr_compiler, ["get_weather", "search"], seq)

    def test_preamble_then_tool_call(self, xgr_compiler) -> None:
        """Commentary preamble (no to=) followed by tool call."""
        seq = [
            # preamble
            V["commentary"],
            V["<|message|>"],
            V["Let me"],
            V[" call"],
            V[" the weather"],
            V["<|end|>"],
            V["<|start|>"],
            V["assistant"],
            V["<|channel|>"],
            # tool call
            V["commentary"],
            V[" to="],
            V["functions."],
            V["get_weather"],
            V["<|message|>"],
            V["{"],
            V['"'],
            V["query"],
            V['"'],
            V[":"],
            V['"'],
            V["hello"],
            V['"'],
            V["}"],
            V["<|end|>"],
            V["<|call|>"],
        ]
        assert _compile_and_run(xgr_compiler, ["get_weather", "search"], seq)

    def test_analysis_preamble_multi_tool(self, xgr_compiler) -> None:
        """Full flow: analysis -> preamble -> tool1 -> tool2."""
        seq = [
            # analysis
            V["analysis"],
            V["<|message|>"],
            V["I need"],
            V[" to check"],
            V["<|end|>"],
            V["<|start|>"],
            V["assistant"],
            V["<|channel|>"],
            # preamble
            V["commentary"],
            V["<|message|>"],
            V["Let me"],
            V[" call"],
            V[" tools"],
            V["<|end|>"],
            V["<|start|>"],
            V["assistant"],
            V["<|channel|>"],
            # tool call 1
            V["commentary"],
            V[" to="],
            V["functions."],
            V["get_weather"],
            V["<|message|>"],
            V["{"],
            V['"'],
            V["location"],
            V['"'],
            V[":"],
            V['"'],
            V["Tokyo"],
            V['"'],
            V["}"],
            V["<|end|>"],
            V["<|call|>"],
            # tool call 2
            V["<|start|>"],
            V["assistant"],
            V["<|channel|>"],
            V["commentary"],
            V[" to="],
            V["functions."],
            V["search"],
            V["<|message|>"],
            V["{"],
            V['"'],
            V["query"],
            V['"'],
            V[":"],
            V['"'],
            V["Tokyo"],
            V['"'],
            V["}"],
            V["<|end|>"],
            V["<|call|>"],
        ]
        assert _compile_and_run(
            xgr_compiler, ["get_weather", "search", "calculate"], seq
        )

    def test_content_with_lt_operator(self, xgr_compiler) -> None:
        """Content containing '<' comparison operator is allowed."""
        seq = [
            V["analysis"],
            V["<|message|>"],
            V["x"],
            V[" < "],  # comparison operator with '<'
            V["x"],
            V["<|end|>"],
            V["<|start|>"],
            V["assistant"],
            V["<|channel|>"],
            V["commentary"],
            V[" to="],
            V["functions."],
            V["get_weather"],
            V["<|message|>"],
            V["{"],
            V["}"],
            V["<|end|>"],
            V["<|call|>"],
        ]
        assert _compile_and_run(xgr_compiler, ["get_weather"], seq)

    def test_content_with_lte_operator(self, xgr_compiler) -> None:
        """Content with '<=' is allowed."""
        seq = [
            V["analysis"],
            V["<|message|>"],
            V["x"],
            V["<="],
            V["x"],
            V["<|end|>"],
            V["<|start|>"],
            V["assistant"],
            V["<|channel|>"],
            V["commentary"],
            V[" to="],
            V["functions."],
            V["search"],
            V["<|message|>"],
            V["{"],
            V["}"],
            V["<|end|>"],
            V["<|call|>"],
        ]
        assert _compile_and_run(xgr_compiler, ["search"], seq)

    def test_content_with_html_tag(self, xgr_compiler) -> None:
        """Content with '</div>' HTML closing tag is allowed."""
        seq = [
            V["analysis"],
            V["<|message|>"],
            V["hello"],
            V["</div>"],
            V["<|end|>"],
            V["<|start|>"],
            V["assistant"],
            V["<|channel|>"],
            V["commentary"],
            V[" to="],
            V["functions."],
            V["get_weather"],
            V["<|message|>"],
            V["{"],
            V["}"],
            V["<|end|>"],
            V["<|call|>"],
        ]
        assert _compile_and_run(xgr_compiler, ["get_weather"], seq)


class TestXgrammarBlocking:
    """Test that the grammar correctly blocks invalid sequences."""

    def test_final_channel_blocked(self, xgr_compiler) -> None:
        """The 'final' channel must be blocked at the channel position."""
        seq = [V["final"]]
        _compile_and_check_blocked(xgr_compiler, ["get_weather"], seq, blocked_at=0)

    def test_final_after_analysis_blocked(self, xgr_compiler) -> None:
        """Final channel is blocked even after an analysis round."""
        seq = [
            V["analysis"],
            V["<|message|>"],
            V["hello"],
            V["<|end|>"],
            V["<|start|>"],
            V["assistant"],
            V["<|channel|>"],
            V["final"],  # should be blocked here
        ]
        _compile_and_check_blocked(xgr_compiler, ["get_weather"], seq, blocked_at=7)

    def test_return_token_blocked(self, xgr_compiler) -> None:
        """<|return|> is never valid in the grammar."""
        grammar = OpenAIToolParser._build_tool_required_grammar(["get_weather"])
        ctx = xgr_compiler.compile_grammar(grammar)
        matcher = xgrammar.GrammarMatcher(ctx)
        # <|return|> should never be accepted at any position
        assert not matcher.accept_token(V["<|return|>"])

    def test_analysis_only_not_terminated(self, xgr_compiler) -> None:
        """Grammar must not terminate after analysis-only (no tool call)."""
        grammar = OpenAIToolParser._build_tool_required_grammar(["get_weather"])
        ctx = xgr_compiler.compile_grammar(grammar)
        matcher = xgrammar.GrammarMatcher(ctx)
        seq = [
            V["analysis"],
            V["<|message|>"],
            V["hello"],
            V["<|end|>"],
        ]
        for tid in seq:
            assert matcher.accept_token(tid)
        assert not matcher.is_terminated()

    def test_wrong_function_name_blocked(self, xgr_compiler) -> None:
        """Tool names not in the grammar are blocked."""
        grammar = OpenAIToolParser._build_tool_required_grammar(["get_weather"])
        ctx = xgr_compiler.compile_grammar(grammar)
        matcher = xgrammar.GrammarMatcher(ctx)
        # Start a tool call
        for tid in [V["commentary"], V[" to="]]:
            assert matcher.accept_token(tid)
        # "functions." accepted
        assert matcher.accept_token(V["functions."])
        # "search" should be blocked (only "get_weather" is in grammar)
        assert not matcher.accept_token(V["search"])


class TestXgrammarBitmask:
    """Test token bitmask at critical positions."""

    def test_channel_position_bitmask(self, xgr_compiler) -> None:
        """After <|channel|>, only 'analysis' and 'commentary' are allowed."""
        grammar = OpenAIToolParser._build_tool_required_grammar(["get_weather"])
        ctx = xgr_compiler.compile_grammar(grammar)
        matcher = xgrammar.GrammarMatcher(ctx)

        vocab = _make_test_vocab()

        # Advance past analysis block to second channel position
        for tid in [
            V["analysis"],
            V["<|message|>"],
            V["hello"],
            V["<|end|>"],
            V["<|start|>"],
            V["assistant"],
            V["<|channel|>"],
        ]:
            assert matcher.accept_token(tid)

        # Check bitmask
        bitmask = xgrammar.allocate_token_bitmask(1, len(vocab))
        matcher.fill_next_token_bitmask(bitmask, 0)

        def is_allowed(token_id: int) -> bool:
            byte_idx = token_id // 32
            bit_idx = token_id % 32
            return bool(bitmask[0, byte_idx].item() & (1 << bit_idx))

        assert is_allowed(V["analysis"]), "analysis should be allowed"
        assert is_allowed(V["commentary"]), "commentary should be allowed"
        assert not is_allowed(V["final"]), "final should be blocked"
        assert not is_allowed(V["<|return|>"]), "<|return|> should be blocked"
