# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Logits processor for forcing specific token sequences in Harmony models.

Used to implement tool_choice="required" by forcing the model to generate
tool call tokens (e.g., "commentary to=") after detecting a trigger sequence
(e.g., "<|end|><|start|>assistant<|channel|>").
"""

from typing import TYPE_CHECKING

import torch

from vllm.logger import init_logger
from vllm.v1.sample.logits_processor.builtin import process_dict_updates
from vllm.v1.sample.logits_processor.interface import (
    BatchUpdate,
    LogitsProcessor,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = init_logger(__name__)

# Key in SamplingParams.extra_args for Harmony tool choice config
HARMONY_TOOL_CHOICE_KEY = "harmony_tool_choice"


class HarmonyToolChoiceLogitsProcessor(LogitsProcessor):
    """Forces specific token sequences after detecting a trigger sequence.

    When the model generates a trigger sequence
    (e.g., <|end|><|start|>assistant<|channel|>), this processor forces
    the next tokens to be the forced sequence (e.g., "commentary to=").

    Configuration is passed via SamplingParams.extra_args:
        extra_args[HARMONY_TOOL_CHOICE_KEY] = {
            "trigger_sequence": [token_id1, token_id2, ...],
            "forced_tokens": [token_id1, token_id2, ...],
        }
    """

    def __init__(
        self, vllm_config: "VllmConfig", device: torch.device, is_pin_memory: bool
    ) -> None:
        self.device = device
        self.pin_memory = is_pin_memory

        # Maps req_index -> (trigger_sequence, forced_tokens, output_tok_ids_ref)
        self.req_configs: dict[int, tuple[list[int], list[int], list[int]]] = {}

        # Maps req_index -> current forcing step (None if not forcing)
        # When forcing, this is the index into forced_tokens for the next token
        self.forcing_state: dict[int, int] = {}

        # Pre-allocated tensor for -inf
        self.neg_inf = torch.tensor(float("-inf"), dtype=torch.float32, device=device)

    def is_argmax_invariant(self) -> bool:
        """This processor changes the argmax by forcing specific tokens."""
        return False

    @staticmethod
    def _extract_config(
        params, prompt_tok_ids: list[int] | None, output_tok_ids: list[int]
    ) -> tuple[list[int], list[int], list[int]] | None:
        """Extract Harmony tool choice config from SamplingParams.extra_args."""
        if not params.extra_args:
            return None

        config = params.extra_args.get(HARMONY_TOOL_CHOICE_KEY)
        if not config:
            return None

        trigger_sequence = config.get("trigger_sequence")
        forced_tokens = config.get("forced_tokens")

        if not trigger_sequence or not forced_tokens:
            logger.warning(
                "Invalid harmony_tool_choice config: "
                "missing trigger_sequence or forced_tokens"
            )
            return None

        return (trigger_sequence, forced_tokens, output_tok_ids)

    def update_state(self, batch_update: BatchUpdate | None) -> None:
        """Update internal state based on batch changes."""
        process_dict_updates(self.req_configs, batch_update, self._extract_config)

        if batch_update:
            # Clean up forcing_state for removed requests
            for index in batch_update.removed:
                self.forcing_state.pop(index, None)

            # Handle moved requests in forcing_state
            for a_idx, b_idx, directionality in batch_update.moved:
                a_state = self.forcing_state.pop(a_idx, None)
                b_state = self.forcing_state.pop(b_idx, None)
                if a_state is not None:
                    self.forcing_state[b_idx] = a_state
                if b_state is not None and directionality.name == "SWAP":
                    self.forcing_state[a_idx] = b_state

    def _check_trigger_sequence(
        self, output_tokens: list[int], trigger_sequence: list[int]
    ) -> bool:
        """Check if the last N tokens match the trigger sequence."""
        seq_len = len(trigger_sequence)
        if len(output_tokens) < seq_len:
            return False
        return list(output_tokens[-seq_len:]) == trigger_sequence

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply token forcing based on trigger sequence detection."""
        if not self.req_configs:
            return logits

        for req_idx, (
            trigger_seq,
            forced_tokens,
            output_tok_ids,
        ) in self.req_configs.items():
            # Check if we're currently in forcing mode
            if req_idx in self.forcing_state:
                step = self.forcing_state[req_idx]
                if step < len(forced_tokens):
                    # Force the next token in the sequence
                    logits[req_idx, :] = self.neg_inf
                    logits[req_idx, forced_tokens[step]] = 0.0
                    self.forcing_state[req_idx] = step + 1
                else:
                    # Forcing complete, return to normal generation
                    del self.forcing_state[req_idx]
                continue

            # Check if trigger sequence was just generated
            if self._check_trigger_sequence(output_tok_ids, trigger_seq):
                # Start forcing mode
                logger.debug(
                    "Trigger sequence detected for request %d, starting token forcing",
                    req_idx,
                )
                self.forcing_state[req_idx] = 0
                # Force the first token
                logits[req_idx, :] = self.neg_inf
                logits[req_idx, forced_tokens[0]] = 0.0
                self.forcing_state[req_idx] = 1

        return logits
