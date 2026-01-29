# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json
from collections.abc import Sequence

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
from vllm.tokenizers import TokenizerLike
from vllm.tool_parsers.abstract_tool_parser import (
    ToolParser,
)
from vllm.v1.sample.logits_processor.harmony_tool_choice import (
    HARMONY_TOOL_CHOICE_KEY,
)

logger = init_logger(__name__)


class OpenAIToolParser(ToolParser):
    """
    Tool parser for GPT-OSS Harmony models.

    Supports tool_choice="required" by using HarmonyToolChoiceLogitsProcessor
    to force tool call generation.
    """

    # Harmony special token names
    HARMONY_END_TOKEN = "<|end|>"
    HARMONY_START_TOKEN = "<|start|>"
    HARMONY_CHANNEL_TOKEN = "<|channel|>"
    HARMONY_RECIPIENT_PREFIX = " to="  # Recipient marker in Harmony format

    def __init__(self, tokenizer: TokenizerLike):
        super().__init__(tokenizer)
        # Cache channel name token IDs
        self._channel_token_ids = self._get_channel_token_ids()

    def _get_channel_token_ids(self) -> dict[str, list[int] | None]:
        """Get channel name token IDs."""
        return {
            "commentary": self._encode_tokens("commentary"),
        }

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

    def _build_harmony_tool_choice_config(self) -> dict | None:
        """
        Build configuration for HarmonyToolChoiceLogitsProcessor.

        This creates a config that forces "commentary to=" after detecting
        the trigger sequence "<|end|><|start|>assistant<|channel|>".

        Returns:
            Config dict with trigger_sequence and forced_tokens, or None if
            required tokens are missing.
        """
        vocab = self.vocab
        end_id = vocab.get(self.HARMONY_END_TOKEN)
        start_id = vocab.get(self.HARMONY_START_TOKEN)
        channel_id = vocab.get(self.HARMONY_CHANNEL_TOKEN)
        assistant_tokens = self._encode_tokens("assistant")

        # Validate required tokens exist
        if (
            end_id is None
            or start_id is None
            or not assistant_tokens
            or channel_id is None
        ):
            logger.warning(
                "Missing Harmony special tokens in vocabulary. "
                "HarmonyToolChoiceLogitsProcessor cannot be configured."
            )
            return None

        # Trigger sequence: <|end|><|start|>assistant<|channel|>
        # Note: assistant may be multiple tokens
        trigger_sequence: list[int] = (
            [end_id, start_id] + assistant_tokens + [channel_id]
        )

        # Forced tokens: "commentary to="
        commentary_tokens = self._channel_token_ids.get("commentary")
        recipient_tokens = self._encode_tokens(self.HARMONY_RECIPIENT_PREFIX)

        if not isinstance(commentary_tokens, list) or not recipient_tokens:
            logger.warning(
                "Could not encode 'commentary to=' tokens. "
                "HarmonyToolChoiceLogitsProcessor cannot be configured."
            )
            return None

        forced_tokens: list[int] = commentary_tokens + recipient_tokens

        logger.debug(
            "HarmonyToolChoiceLogitsProcessor config: "
            "trigger_sequence=%s, forced_tokens=%s",
            trigger_sequence,
            forced_tokens,
        )

        return {
            "trigger_sequence": trigger_sequence,
            "forced_tokens": forced_tokens,
        }

    def adjust_request(self, request: ChatCompletionRequest) -> ChatCompletionRequest:
        """
        Adjust request for GPT-OSS tool_choice="required" support.

        For tool_choice="required", configures HarmonyToolChoiceLogitsProcessor
        to force tool call generation by:
        1. Detecting trigger sequence: <|end|><|start|>assistant<|channel|>
        2. Forcing tokens: "commentary to="
        """
        if not request.tools:
            return request

        # For tool_choice != "required", use default behavior
        if request.tool_choice != "required":
            return super().adjust_request(request)

        logger.debug("GPT-OSS tool_choice=required: configuring token forcing")

        # Build HarmonyToolChoiceLogitsProcessor config
        tool_choice_config = self._build_harmony_tool_choice_config()
        if tool_choice_config:
            # Store in extra_args for HarmonyToolChoiceLogitsProcessor
            request._tool_parser_extra_args = {  # type: ignore[attr-defined]
                HARMONY_TOOL_CHOICE_KEY: tool_choice_config
            }
            logger.debug(
                "Configured HarmonyToolChoiceLogitsProcessor for tool_choice=required"
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
                "OpenAIToolParser requires token IDs and does not support "
                "text-based extraction."
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

        # Extract partial content from the parser state if generation was truncated
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
        raise NotImplementedError("Not being used, manual parsing in serving.py")
