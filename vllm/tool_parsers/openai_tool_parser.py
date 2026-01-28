# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json
from collections.abc import Sequence
from typing import TYPE_CHECKING

from vllm.entrypoints.openai.parser.harmony_utils import parse_output_into_messages
from vllm.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    DeltaMessage,
    ExtractedToolCallInformation,
    FunctionCall,
    ToolCall,
)
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
_END_TOKEN_ID = 200007       # <|end|>
_START_TOKEN_ID = 200006     # <|start|>
_ASSISTANT_TOKEN_ID = 173781 # assistant
_CHANNEL_TOKEN_ID = 200005   # <|channel|>
_MESSAGE_TOKEN_ID = 200008   # <|message|>


class OpenAIToolParser(ToolParser):
    def __init__(self, tokenizer: "TokenizerLike"):
        super().__init__(tokenizer)
        # Pre-compute token IDs for channel names
        self._final_token_id = self._encode_single_token("final")
        self._analysis_token_id = self._encode_single_token("analysis")
        self._commentary_token_ids = self._encode_tokens("commentary")

    def _encode_single_token(self, text: str) -> int | None:
        """Encode text and return token ID if it's a single token."""
        try:
            ids = self.model_tokenizer.encode(text, add_special_tokens=False)
            return ids[0] if len(ids) == 1 else None
        except Exception:
            return None

    def _encode_tokens(self, text: str) -> list[int] | None:
        """Encode text and return all token IDs."""
        try:
            ids = self.model_tokenizer.encode(text, add_special_tokens=False)
            return ids if ids else None
        except Exception:
            return None

    def adjust_request(
        self, request: ChatCompletionRequest
    ) -> ChatCompletionRequest:
        """
        Override adjust_request for GPT-OSS tool_choice="required".

        Use bad_words token sequences to block non-tool-call paths:
        - Block <|end|><|start|>assistant<|channel|>final (new message to final)
        - Block <|end|><|start|>assistant<|channel|>analysis (new message to analysis)
        - Block commentary<|message|> (commentary without recipient)

        This allows first message to use analysis channel for reasoning,
        but forces subsequent messages to use commentary with recipient (tool call).
        """
        if not request.tools:
            return request

        # For tool_choice != "required", use default behavior
        if request.tool_choice != "required":
            return super().adjust_request(request)

        # For tool_choice="required", use bad_words approach
        logger.debug("GPT-OSS tool_choice=required: using bad_words approach")

        # Build bad_words token sequences
        bad_sequences: list[list[int]] = []

        # Common prefix for new assistant message
        new_msg_prefix = [
            _END_TOKEN_ID,      # <|end|>
            _START_TOKEN_ID,    # <|start|>
            _ASSISTANT_TOKEN_ID,  # assistant
            _CHANNEL_TOKEN_ID,  # <|channel|>
        ]

        # Block <|end|><|start|>assistant<|channel|>final
        if self._final_token_id is not None:
            seq = new_msg_prefix + [self._final_token_id]
            bad_sequences.append(seq)
            logger.debug(f"Blocking new message to final: {seq}")

        # Block <|end|><|start|>assistant<|channel|>analysis
        if self._analysis_token_id is not None:
            seq = new_msg_prefix + [self._analysis_token_id]
            bad_sequences.append(seq)
            logger.debug(f"Blocking new message to analysis: {seq}")

        # Block commentary<|message|> (without recipient)
        if self._commentary_token_ids is not None:
            seq = self._commentary_token_ids + [_MESSAGE_TOKEN_ID]
            bad_sequences.append(seq)
            logger.debug(f"Blocking commentary without recipient: {seq}")

        # Store in vllm_xargs for later application to SamplingParams
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
