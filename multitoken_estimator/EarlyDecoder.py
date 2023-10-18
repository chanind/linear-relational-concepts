from typing import Iterable

import torch
from tokenizers import Tokenizer
from torch import nn

from multitoken_estimator.torch_utils import get_module


class EarlyDecoder:
    model: nn.Module
    tokenizer: Tokenizer
    decoding_layers: tuple[str, ...]

    def __init__(
        self, model: nn.Module, tokenizer: Tokenizer, decoding_layers: Iterable[str]
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.decoding_layers = tuple(decoding_layers)

    @torch.no_grad()
    def decode(self, activation: torch.Tensor, topk: int = 10) -> list[str]:
        """
        Decode the input using the model's decoder.
        """
        decoded_logits = activation
        for layer in self.decoding_layers:
            layer_module = get_module(self.model, layer)
            decoded_logits = layer_module(decoded_logits)
        return [self.tokenizer.decode(val) for val in decoded_logits.topk(topk).indices]
