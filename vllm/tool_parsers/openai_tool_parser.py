# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json
from collections.abc import Sequence
from typing import TYPE_CHECKING

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
)
from vllm.entrypoints.openai.engine.protocol import (
    DeltaMessage,
    ExtractedToolCallInformation,
    FunctionCall,
    ToolCall,
)
from vllm.entrypoints.openai.parser.harmony_utils import parse_output_into_messages
from vllm.logger import init_logger
from vllm.tool_parsers.abstract_tool_parser import (
    ToolParser,
)

if TYPE_CHECKING:
    from vllm.tokenizers import TokenizerLike
else:
    TokenizerLike = object

logger = init_logger(__name__)


# GPT-OSS Harmony special token IDs
_CHANNEL_TOKEN_ID = 200005   # <|channel|>
_MESSAGE_TOKEN_ID = 200008   # <|message|>


class OpenAIToolParser(ToolParser):
    def __init__(self, tokenizer: "TokenizerLike"):
        super().__init__(tokenizer)
        # Pre-compute token IDs for channel names
        self._final_token_id = self._encode_single_token("final")
        self._analysis_token_id = self._encode_single_token("analysis")
        self._commentary_token_id = self._encode_single_token("commentary")

    def _encode_single_token(self, text: str) -> int | None:
        """Encode text and return token ID if it's a single token."""
        try:
            ids = self.model_tokenizer.encode(text, add_special_tokens=False)
            return ids[0] if len(ids) == 1 else None
        except Exception:
            return None

    def adjust_request(
        self, request: ChatCompletionRequest
    ) -> ChatCompletionRequest:
        """
        Override adjust_request for GPT-OSS tool_choice="required".

        Instead of using JSON schema bitmask (which doesn't work well with
        Harmony format), we use bad_words token sequences to block
        non-tool-call paths:
        - Block <|channel|>final (direct response)
        - Block <|channel|>analysis (reasoning only)
        - Block commentary<|message|> (preamble without tool call)
        """
        if not request.tools:
            return request

        # For tool_choice != "required", use default behavior (JSON schema)
        if request.tool_choice != "required":
            return super().adjust_request(request)

        # For tool_choice="required", use bad_words approach
        logger.debug("GPT-OSS tool_choice=required: using bad_words approach")

        # 1. Disable JSON schema (bitmask) - GPT-OSS uses Harmony format
        request.structured_outputs = None
        request.response_format = None

        # 2. Build bad_words token sequences
        bad_sequences: list[list[int]] = []

        # Block <|channel|>final
        if self._final_token_id is not None:
            bad_sequences.append([_CHANNEL_TOKEN_ID, self._final_token_id])
            logger.debug(
                f"Blocking sequence: [<|channel|>, final] = "
                f"[{_CHANNEL_TOKEN_ID}, {self._final_token_id}]"
            )

        # Block <|channel|>analysis
        if self._analysis_token_id is not None:
            bad_sequences.append([_CHANNEL_TOKEN_ID, self._analysis_token_id])
            logger.debug(
                f"Blocking sequence: [<|channel|>, analysis] = "
                f"[{_CHANNEL_TOKEN_ID}, {self._analysis_token_id}]"
            )

        # Block commentary<|message|> (preamble without recipient)
        if self._commentary_token_id is not None:
            bad_sequences.append([self._commentary_token_id, _MESSAGE_TOKEN_ID])
            logger.debug(
                f"Blocking sequence: [commentary, <|message|>] = "
                f"[{self._commentary_token_id}, {_MESSAGE_TOKEN_ID}]"
            )

        # 3. Store in vllm_xargs for later application to SamplingParams
        if bad_sequences:
            if request.vllm_xargs is None:
                request.vllm_xargs = {}
            request.vllm_xargs["_gptoss_bad_words_token_ids"] = bad_sequences
            logger.debug(
                f"Stored {len(bad_sequences)} bad_words sequences in vllm_xargs"
            )

        return request

    def extract_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest,
        token_ids: Sequence[int] | None = None,
    ) -> ExtractedToolCallInformation:
        if token_ids is None:
            raise NotImplementedError(
                "OpenAIToolParser requires token IDs and does not support text-based extraction."  # noqa: E501
            )

        parser = parse_output_into_messages(token_ids)
        tool_calls = []
        final_content = None
        commentary_content = None

        if len(parser.messages) > 0:
            for msg in parser.messages:
                if len(msg.content) < 1:
                    continue
                msg_text = msg.content[0].text
                if msg.recipient and msg.recipient.startswith("functions."):
                    # If no content-type is given assume JSON, as that's the
                    # most common case with gpt-oss models.
                    if not msg.content_type or "json" in msg.content_type:
                        # load and dump the JSON text to check validity and
                        # remove any extra newlines or other odd formatting
                        try:
                            tool_args = json.dumps(json.loads(msg_text))
                        except json.JSONDecodeError:
                            logger.exception(
                                "Error decoding JSON tool call from response."
                            )
                            tool_args = msg_text
                    else:
                        tool_args = msg_text
                    tool_calls.append(
                        ToolCall(
                            type="function",
                            function=FunctionCall(
                                name=msg.recipient.split("functions.")[1],
                                arguments=tool_args,
                            ),
                        )
                    )
                elif msg.channel == "final":
                    final_content = msg_text
                elif msg.channel == "commentary" and not msg.recipient:
                    commentary_content = msg_text

        # Extract partial content from the parser state if the generation was truncated
        if parser.current_content:
            if parser.current_channel == "final":
                final_content = parser.current_content
            elif (
                parser.current_channel == "commentary" and not parser.current_recipient
            ):
                commentary_content = parser.current_content

        return ExtractedToolCallInformation(
            tools_called=len(tool_calls) > 0,
            tool_calls=tool_calls,
            # prefer final content over commentary content if both are present
            # commentary content is tool call preambles meant to be shown to the user
            content=final_content or commentary_content,
        )

    def extract_tool_calls_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
        request: ChatCompletionRequest,
    ) -> DeltaMessage | None:
        raise NotImplementedError(
            "Not being used, manual parsing in serving_chat.py"  # noqa: E501
        )
