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


def test_build_grammar_many_tools(parser: OpenAIToolParser) -> None:
    """Grammar with 20 tools should contain all alternatives."""
    tools = [f"tool_{i}" for i in range(20)]
    grammar = parser._build_tool_required_grammar(tools)
    for name in tools:
        assert f'"functions.{name}"' in grammar


def test_build_grammar_tool_names_with_numbers_underscores(
    parser: OpenAIToolParser,
) -> None:
    """Tool names with numbers and underscores are preserved."""
    grammar = parser._build_tool_required_grammar(["get_weather_v2", "search_123"])
    assert '"functions.get_weather_v2"' in grammar
    assert '"functions.search_123"' in grammar


def test_build_grammar_rejects_tool_name_with_quotes(
    parser: OpenAIToolParser,
) -> None:
    """Tool names containing quotes must be rejected to prevent grammar injection."""
    with pytest.raises(ValueError, match="invalid for EBNF grammar"):
        parser._build_tool_required_grammar(['get"weather'])


def test_build_grammar_rejects_tool_name_with_newlines(
    parser: OpenAIToolParser,
) -> None:
    """Tool names containing newlines must be rejected."""
    with pytest.raises(ValueError, match="invalid for EBNF grammar"):
        parser._build_tool_required_grammar(["get\nweather"])


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


def test_adjust_request_tool_choice_none(parser: OpenAIToolParser) -> None:
    """tool_choice='none' should not activate grammar."""
    request = ChatCompletionRequest(
        model="test",
        messages=[{"role": "user", "content": "hi"}],
        tools=_make_tools("f"),
        tool_choice="none",
    )
    result = parser.adjust_request(request)
    assert result.structured_outputs is None


def test_adjust_request_with_tools_default_choice(
    parser: OpenAIToolParser,
) -> None:
    """tools present but tool_choice not set should not activate grammar."""
    request = ChatCompletionRequest(
        model="test",
        messages=[{"role": "user", "content": "hi"}],
        tools=_make_tools("get_weather"),
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
        "\n",  # 40 — newline
        "[",  # 41
        "]",  # 42
        "search_web",  # 43 — prefix edge case
        " finally",  # 44 — must be blocked at channel position
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


def _bitmask_allowed(bitmask, token_id: int) -> bool:
    """Check if a token is allowed by the xgrammar bitmask."""
    byte_idx = token_id // 32
    bit_idx = token_id % 32
    return bool(bitmask[0, byte_idx].item() & (1 << bit_idx))


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


# ---------------------------------------------------------------------------
# Termination & EOS tests — critical for production decoding
# ---------------------------------------------------------------------------


class TestXgrammarTermination:
    """Verify grammar terminates at correct points.

    In production, xgrammar blocks EOS until the grammar is satisfied.
    If termination is wrong, the model either hangs (never EOS) or
    stops prematurely (EOS before tool call).
    """

    def test_satisfied_after_single_tool_call(self, xgr_compiler) -> None:
        """After one tool_block, EOS is allowed and more tools can start.

        Note: is_terminated() is False because more_tool* can accept more.
        The correct production check is that EOS is allowed in bitmask.
        """
        grammar = OpenAIToolParser._build_tool_required_grammar(["get_weather"])
        ctx = xgr_compiler.compile_grammar(grammar)
        matcher = xgrammar.GrammarMatcher(ctx)
        seq = [
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
        for tid in seq:
            assert matcher.accept_token(tid)
        # Grammar is NOT terminated (more_tool* can match)
        assert not matcher.is_terminated()
        # But EOS IS allowed — grammar is satisfied
        bitmask = xgrammar.allocate_token_bitmask(1, len(VOCAB))
        matcher.fill_next_token_bitmask(bitmask, 0)
        assert _bitmask_allowed(bitmask, V["<|eos|>"]), (
            "EOS must be allowed after complete tool call"
        )
        # More tool calls can also start
        assert _bitmask_allowed(bitmask, V["<|start|>"]), (
            "more_tool path must remain open"
        )

    def test_satisfied_after_multi_tool_calls(self, xgr_compiler) -> None:
        """After tool_block + more_tool, EOS is allowed."""
        grammar = OpenAIToolParser._build_tool_required_grammar(
            ["get_weather", "search"]
        )
        ctx = xgr_compiler.compile_grammar(grammar)
        matcher = xgrammar.GrammarMatcher(ctx)
        seq = [
            # tool 1
            V["commentary"],
            V[" to="],
            V["functions."],
            V["get_weather"],
            V["<|message|>"],
            V["{"],
            V["}"],
            V["<|end|>"],
            V["<|call|>"],
            # tool 2
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
        for tid in seq:
            assert matcher.accept_token(tid)
        bitmask = xgrammar.allocate_token_bitmask(1, len(VOCAB))
        matcher.fill_next_token_bitmask(bitmask, 0)
        assert _bitmask_allowed(bitmask, V["<|eos|>"]), (
            "EOS must be allowed after multiple tool calls"
        )

    def test_not_terminated_mid_tool_call(self, xgr_compiler) -> None:
        """Not terminated when <|call|> has not been emitted yet."""
        grammar = OpenAIToolParser._build_tool_required_grammar(["get_weather"])
        ctx = xgr_compiler.compile_grammar(grammar)
        matcher = xgrammar.GrammarMatcher(ctx)
        seq = [
            V["commentary"],
            V[" to="],
            V["functions."],
            V["get_weather"],
            V["<|message|>"],
            V["{"],
            V["}"],
            V["<|end|>"],
            # missing <|call|>
        ]
        for tid in seq:
            assert matcher.accept_token(tid)
        assert not matcher.is_terminated()

    def test_eos_blocked_before_any_tool_call(self, xgr_compiler) -> None:
        """EOS must be blocked in bitmask when no tool call has been made."""
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
        bitmask = xgrammar.allocate_token_bitmask(1, len(VOCAB))
        matcher.fill_next_token_bitmask(bitmask, 0)
        assert not _bitmask_allowed(bitmask, V["<|eos|>"]), (
            "EOS must be blocked before tool call"
        )

    def test_eos_allowed_after_complete_tool_call(self, xgr_compiler) -> None:
        """EOS must be allowed in bitmask after grammar is satisfied."""
        grammar = OpenAIToolParser._build_tool_required_grammar(["get_weather"])
        ctx = xgr_compiler.compile_grammar(grammar)
        matcher = xgrammar.GrammarMatcher(ctx)
        seq = [
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
        for tid in seq:
            assert matcher.accept_token(tid)
        bitmask = xgrammar.allocate_token_bitmask(1, len(VOCAB))
        matcher.fill_next_token_bitmask(bitmask, 0)
        assert _bitmask_allowed(bitmask, V["<|eos|>"]), (
            "EOS must be allowed after tool call"
        )


# ---------------------------------------------------------------------------
# Empty content tests
# ---------------------------------------------------------------------------


class TestXgrammarEmptyContent:
    """Test empty content in various block types.

    Models sometimes produce empty analysis or empty tool arguments.
    The content rule uses *, so zero-length content must be valid.
    """

    def test_empty_tool_call_arguments(self, xgr_compiler) -> None:
        """Tool call with nothing between <|message|> and <|end|>."""
        seq = [
            V["commentary"],
            V[" to="],
            V["functions."],
            V["get_weather"],
            V["<|message|>"],
            V["<|end|>"],
            V["<|call|>"],
        ]
        assert _compile_and_run(xgr_compiler, ["get_weather"], seq)

    def test_empty_analysis_content(self, xgr_compiler) -> None:
        """Analysis block with empty message body."""
        seq = [
            V["analysis"],
            V["<|message|>"],
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

    def test_empty_preamble_content(self, xgr_compiler) -> None:
        """Commentary preamble with empty message body."""
        seq = [
            V["commentary"],
            V["<|message|>"],
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


# ---------------------------------------------------------------------------
# Commentary disambiguation — CRITICAL for production
# ---------------------------------------------------------------------------


class TestXgrammarCommentaryDisambiguation:
    """Verify 'commentary' correctly branches to preamble vs tool call.

    The grammar has an ambiguity at 'commentary':
    - non_tool_block: 'commentary' followed by '<|message|>'
    - tool_block: 'commentary to=' followed by func_name

    xgrammar must keep both paths open until the next token resolves it.
    If this fails, the model either can't make preambles or can't make
    tool calls — catastrophic in production.
    """

    def test_after_commentary_both_paths_available(self, xgr_compiler) -> None:
        """After 'commentary', both '<|message|>' and ' to=' are allowed."""
        grammar = OpenAIToolParser._build_tool_required_grammar(["get_weather"])
        ctx = xgr_compiler.compile_grammar(grammar)
        matcher = xgrammar.GrammarMatcher(ctx)
        assert matcher.accept_token(V["commentary"])

        bitmask = xgrammar.allocate_token_bitmask(1, len(VOCAB))
        matcher.fill_next_token_bitmask(bitmask, 0)
        assert _bitmask_allowed(bitmask, V["<|message|>"]), "preamble path must be open"
        assert _bitmask_allowed(bitmask, V[" to="]), "tool call path must be open"
        assert not _bitmask_allowed(bitmask, V["final"])
        assert not _bitmask_allowed(bitmask, V["hello"])

    def test_disambiguation_in_more_tool(self, xgr_compiler) -> None:
        """After first tool + <|channel|> + commentary, both paths open."""
        grammar = OpenAIToolParser._build_tool_required_grammar(["get_weather"])
        ctx = xgr_compiler.compile_grammar(grammar)
        matcher = xgrammar.GrammarMatcher(ctx)
        seq = [
            V["commentary"],
            V[" to="],
            V["functions."],
            V["get_weather"],
            V["<|message|>"],
            V["{"],
            V["}"],
            V["<|end|>"],
            V["<|call|>"],
            # more_tool header
            V["<|start|>"],
            V["assistant"],
            V["<|channel|>"],
            V["commentary"],
        ]
        for tid in seq:
            assert matcher.accept_token(tid)

        bitmask = xgrammar.allocate_token_bitmask(1, len(VOCAB))
        matcher.fill_next_token_bitmask(bitmask, 0)
        assert _bitmask_allowed(bitmask, V["<|message|>"]), (
            "preamble path must be open in more_tool"
        )
        assert _bitmask_allowed(bitmask, V[" to="]), (
            "tool call path must be open in more_tool"
        )


# ---------------------------------------------------------------------------
# Special tokens blocked inside content
# ---------------------------------------------------------------------------


class TestXgrammarContentSpecialTokens:
    """Verify all Harmony special tokens are blocked inside content.

    The content rule ([^<] | '<' [^|])* must block <|...|> tokens
    while allowing regular '<' usage. This is the core safety
    mechanism preventing the model from injecting channels or
    control tokens inside message bodies.
    """

    def test_all_special_tokens_blocked_in_content(self, xgr_compiler) -> None:
        """Every <|...|> token must be blocked inside content."""
        grammar = OpenAIToolParser._build_tool_required_grammar(["get_weather"])
        ctx = xgr_compiler.compile_grammar(grammar)
        matcher = xgrammar.GrammarMatcher(ctx)
        # Navigate to content position
        for tid in [V["analysis"], V["<|message|>"], V["hello"]]:
            assert matcher.accept_token(tid)

        bitmask = xgrammar.allocate_token_bitmask(1, len(VOCAB))
        matcher.fill_next_token_bitmask(bitmask, 0)
        # All Harmony specials blocked
        assert not _bitmask_allowed(bitmask, V["<|start|>"])
        assert not _bitmask_allowed(bitmask, V["<|channel|>"])
        assert not _bitmask_allowed(bitmask, V["<|call|>"])
        assert not _bitmask_allowed(bitmask, V["<|return|>"])
        assert not _bitmask_allowed(bitmask, V["<|message|>"])
        # <|end|> IS allowed (it terminates content)
        assert _bitmask_allowed(bitmask, V["<|end|>"])
        # Regular content tokens remain allowed
        assert _bitmask_allowed(bitmask, V["hello"])
        assert _bitmask_allowed(bitmask, V[" < "])
        assert _bitmask_allowed(bitmask, V["<="])
        assert _bitmask_allowed(bitmask, V["</div>"])


# ---------------------------------------------------------------------------
# Tool name edge cases
# ---------------------------------------------------------------------------


class TestXgrammarToolNameEdgeCases:
    """Test edge cases around tool/function names."""

    def test_tool_name_prefix_of_another(self, xgr_compiler) -> None:
        """Both 'search' and 'search_web' in grammar; call search_web."""
        seq = [
            V["commentary"],
            V[" to="],
            V["functions."],
            V["search_web"],
            V["<|message|>"],
            V["{"],
            V["}"],
            V["<|end|>"],
            V["<|call|>"],
        ]
        assert _compile_and_run(xgr_compiler, ["search", "search_web"], seq)

    def test_prefix_name_also_works(self, xgr_compiler) -> None:
        """When both 'search' and 'search_web' exist, 'search' is valid."""
        seq = [
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
        assert _compile_and_run(xgr_compiler, ["search", "search_web"], seq)

    def test_same_tool_called_twice(self, xgr_compiler) -> None:
        """Same tool called in consecutive tool_blocks."""
        seq = [
            # first call
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
            # second call — same tool, different args
            V["<|start|>"],
            V["assistant"],
            V["<|channel|>"],
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


# ---------------------------------------------------------------------------
# Sequence ordering edge cases
# ---------------------------------------------------------------------------


class TestXgrammarSequenceOrdering:
    """Test atypical but valid orderings of non_tool_blocks."""

    def test_commentary_then_analysis_then_tool(self, xgr_compiler) -> None:
        """Commentary preamble -> analysis -> tool call (reversed order)."""
        seq = [
            # commentary preamble first
            V["commentary"],
            V["<|message|>"],
            V["Let me"],
            V[" call"],
            V["<|end|>"],
            V["<|start|>"],
            V["assistant"],
            V["<|channel|>"],
            # then analysis
            V["analysis"],
            V["<|message|>"],
            V["I need"],
            V[" to check"],
            V["<|end|>"],
            V["<|start|>"],
            V["assistant"],
            V["<|channel|>"],
            # then tool call
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

    def test_analysis_between_tool_calls(self, xgr_compiler) -> None:
        """Tool call -> analysis -> second tool call (more_tool path)."""
        seq = [
            # tool call 1
            V["commentary"],
            V[" to="],
            V["functions."],
            V["get_weather"],
            V["<|message|>"],
            V["{"],
            V["}"],
            V["<|end|>"],
            V["<|call|>"],
            # analysis between tools (inside more_tool)
            V["<|start|>"],
            V["assistant"],
            V["<|channel|>"],
            V["analysis"],
            V["<|message|>"],
            V["I need"],
            V[" to check"],
            V["<|end|>"],
            V["<|start|>"],
            V["assistant"],
            V["<|channel|>"],
            # tool call 2
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


# ---------------------------------------------------------------------------
# Advanced blocking tests
# ---------------------------------------------------------------------------


class TestXgrammarBlockingAdvanced:
    """Advanced blocking tests for production edge cases."""

    def test_start_token_rejected_at_root(self, xgr_compiler) -> None:
        """<|start|> must not be accepted at the very beginning."""
        grammar = OpenAIToolParser._build_tool_required_grammar(["get_weather"])
        ctx = xgr_compiler.compile_grammar(grammar)
        matcher = xgrammar.GrammarMatcher(ctx)
        assert not matcher.accept_token(V["<|start|>"])

    def test_message_token_rejected_at_root(self, xgr_compiler) -> None:
        """<|message|> must not be accepted at root."""
        grammar = OpenAIToolParser._build_tool_required_grammar(["get_weather"])
        ctx = xgr_compiler.compile_grammar(grammar)
        matcher = xgrammar.GrammarMatcher(ctx)
        assert not matcher.accept_token(V["<|message|>"])

    def test_arbitrary_text_rejected_at_root(self, xgr_compiler) -> None:
        """Plain text like 'hello' must not be accepted at root."""
        grammar = OpenAIToolParser._build_tool_required_grammar(["get_weather"])
        ctx = xgr_compiler.compile_grammar(grammar)
        matcher = xgrammar.GrammarMatcher(ctx)
        assert not matcher.accept_token(V["hello"])

    def test_finally_blocked_at_channel_position(self, xgr_compiler) -> None:
        """' finally' (space+finally) must be blocked at channel pos."""
        seq = [
            V["analysis"],
            V["<|message|>"],
            V["hello"],
            V["<|end|>"],
            V["<|start|>"],
            V["assistant"],
            V["<|channel|>"],
            V[" finally"],  # blocked
        ]
        _compile_and_check_blocked(xgr_compiler, ["get_weather"], seq, blocked_at=7)

    def test_tool_call_without_call_not_terminated(self, xgr_compiler) -> None:
        """Tool block without trailing <|call|> must not terminate."""
        grammar = OpenAIToolParser._build_tool_required_grammar(["get_weather"])
        ctx = xgr_compiler.compile_grammar(grammar)
        matcher = xgrammar.GrammarMatcher(ctx)
        seq = [
            V["commentary"],
            V[" to="],
            V["functions."],
            V["get_weather"],
            V["<|message|>"],
            V["{"],
            V["}"],
            V["<|end|>"],
        ]
        for tid in seq:
            assert matcher.accept_token(tid)
        assert not matcher.is_terminated()

    def test_root_bitmask_only_allows_channels(self, xgr_compiler) -> None:
        """At root position, only 'analysis' and 'commentary' are valid."""
        grammar = OpenAIToolParser._build_tool_required_grammar(["get_weather"])
        ctx = xgr_compiler.compile_grammar(grammar)
        matcher = xgrammar.GrammarMatcher(ctx)

        bitmask = xgrammar.allocate_token_bitmask(1, len(VOCAB))
        matcher.fill_next_token_bitmask(bitmask, 0)
        assert _bitmask_allowed(bitmask, V["analysis"])
        assert _bitmask_allowed(bitmask, V["commentary"])
        assert not _bitmask_allowed(bitmask, V["final"])
        assert not _bitmask_allowed(bitmask, V["<|start|>"])
        assert not _bitmask_allowed(bitmask, V["<|message|>"])
        assert not _bitmask_allowed(bitmask, V["<|return|>"])
        assert not _bitmask_allowed(bitmask, V["hello"])
        assert not _bitmask_allowed(bitmask, V["<|eos|>"])


# ---------------------------------------------------------------------------
# Content edge cases
# ---------------------------------------------------------------------------


class TestXgrammarContentEdgeCases:
    """Test content patterns that could trip up the grammar."""

    def test_content_with_newlines(self, xgr_compiler) -> None:
        """Content containing newline characters."""
        seq = [
            V["analysis"],
            V["<|message|>"],
            V["hello"],
            V["\n"],
            V["world"],
            V["\n"],
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

    def test_content_with_json_structure(self, xgr_compiler) -> None:
        """Analysis content that looks like JSON."""
        seq = [
            V["analysis"],
            V["<|message|>"],
            V["{"],
            V['"'],
            V["query"],
            V['"'],
            V[":"],
            V["["],
            V['"'],
            V["hello"],
            V['"'],
            V[","],
            V['"'],
            V["world"],
            V['"'],
            V["]"],
            V["}"],
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

    def test_content_with_multiple_lt_chars(self, xgr_compiler) -> None:
        """Content with multiple '<' characters in sequence."""
        seq = [
            V["analysis"],
            V["<|message|>"],
            V["x"],
            V[" < "],
            V["x"],
            V["<="],
            V["x"],
            V[" < "],
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


# ---------------------------------------------------------------------------
# Production gap tests — final sweep
# ---------------------------------------------------------------------------


class TestXgrammarEOSMidSequence:
    """Verify EOS is blocked at every critical mid-sequence point."""

    def test_eos_blocked_mid_tool_arguments(self, xgr_compiler) -> None:
        """EOS must be blocked while model is generating tool arguments."""
        grammar = OpenAIToolParser._build_tool_required_grammar(["get_weather"])
        ctx = xgr_compiler.compile_grammar(grammar)
        matcher = xgrammar.GrammarMatcher(ctx)
        # Start tool call, get into content position
        seq = [
            V["commentary"],
            V[" to="],
            V["functions."],
            V["get_weather"],
            V["<|message|>"],
            V["{"],
        ]
        for tid in seq:
            assert matcher.accept_token(tid)
        bitmask = xgrammar.allocate_token_bitmask(1, len(VOCAB))
        matcher.fill_next_token_bitmask(bitmask, 0)
        assert not _bitmask_allowed(bitmask, V["<|eos|>"]), (
            "EOS must be blocked mid-tool-arguments"
        )

    def test_eos_blocked_after_multiple_analysis_rounds(self, xgr_compiler) -> None:
        """EOS stays blocked even after many analysis rounds (no tool)."""
        grammar = OpenAIToolParser._build_tool_required_grammar(["get_weather"])
        ctx = xgr_compiler.compile_grammar(grammar)
        matcher = xgrammar.GrammarMatcher(ctx)
        # 3 rounds of analysis, no tool call
        for _ in range(3):
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
        bitmask = xgrammar.allocate_token_bitmask(1, len(VOCAB))
        matcher.fill_next_token_bitmask(bitmask, 0)
        assert not _bitmask_allowed(bitmask, V["<|eos|>"]), (
            "EOS must stay blocked without any tool call"
        )


class TestXgrammarMoreToolVariations:
    """Test more_tool paths not yet covered."""

    def test_commentary_preamble_between_tool_calls(self, xgr_compiler) -> None:
        """Commentary preamble (not analysis) between two tool calls."""
        seq = [
            # tool call 1
            V["commentary"],
            V[" to="],
            V["functions."],
            V["get_weather"],
            V["<|message|>"],
            V["{"],
            V["}"],
            V["<|end|>"],
            V["<|call|>"],
            # commentary preamble between tools
            V["<|start|>"],
            V["assistant"],
            V["<|channel|>"],
            V["commentary"],
            V["<|message|>"],
            V["Let me"],
            V[" call"],
            V[" tools"],
            V["<|end|>"],
            V["<|start|>"],
            V["assistant"],
            V["<|channel|>"],
            # tool call 2
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

    def test_three_consecutive_tool_calls(self, xgr_compiler) -> None:
        """Three tool calls in sequence (more_tool* with count=2)."""
        tool_names = ["get_weather", "search", "calculate"]
        parts: list[int] = []
        for i, name in enumerate(tool_names):
            if i > 0:
                parts.extend(
                    [
                        V["<|start|>"],
                        V["assistant"],
                        V["<|channel|>"],
                    ]
                )
            parts.extend(
                [
                    V["commentary"],
                    V[" to="],
                    V["functions."],
                    V[name],
                    V["<|message|>"],
                    V["{"],
                    V["}"],
                    V["<|end|>"],
                    V["<|call|>"],
                ]
            )
        assert _compile_and_run(xgr_compiler, tool_names, parts)


class TestXgrammarContentAmbiguity:
    """Verify grammar keywords are treated as plain text inside content."""

    def test_channel_names_in_content(self, xgr_compiler) -> None:
        """'analysis' and 'commentary' tokens in content are text."""
        seq = [
            V["analysis"],
            V["<|message|>"],
            # content contains "analysis"/"commentary" as plain text
            V["analysis"],
            V["commentary"],
            V["hello"],
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

    def test_to_equals_token_in_content(self, xgr_compiler) -> None:
        """' to=' token inside content (after <|message|>) is text."""
        seq = [
            V["analysis"],
            V["<|message|>"],
            V["hello"],
            V[" to="],  # plain text, not routing syntax
            V["world"],
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

    def test_functions_dot_in_content(self, xgr_compiler) -> None:
        """'functions.' token inside content is plain text."""
        seq = [
            V["analysis"],
            V["<|message|>"],
            V["hello"],
            V["functions."],
            V["world"],
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

    def test_assistant_token_in_content(self, xgr_compiler) -> None:
        """'assistant' token inside content is plain text."""
        seq = [
            V["analysis"],
            V["<|message|>"],
            V["assistant"],
            V["hello"],
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


class TestXgrammarPositionalBitmask:
    """Verify bitmask constraints at specific grammar positions."""

    def test_after_func_name_only_message_allowed(self, xgr_compiler) -> None:
        """After func_name, only '<|message|>' should be allowed."""
        grammar = OpenAIToolParser._build_tool_required_grammar(["get_weather"])
        ctx = xgr_compiler.compile_grammar(grammar)
        matcher = xgrammar.GrammarMatcher(ctx)
        for tid in [
            V["commentary"],
            V[" to="],
            V["functions."],
            V["get_weather"],
        ]:
            assert matcher.accept_token(tid)

        bitmask = xgrammar.allocate_token_bitmask(1, len(VOCAB))
        matcher.fill_next_token_bitmask(bitmask, 0)
        assert _bitmask_allowed(bitmask, V["<|message|>"])
        assert not _bitmask_allowed(bitmask, V["<|end|>"])
        assert not _bitmask_allowed(bitmask, V["<|call|>"])
        assert not _bitmask_allowed(bitmask, V["hello"])
        assert not _bitmask_allowed(bitmask, V["<|eos|>"])
