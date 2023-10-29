from typing import List

import numpy as np
import torch
from torch import Tensor

from hw_asr.text_encoder import BaseTextEncoder

from .base_metric import BaseMetric
from .utils import calc_wer


class ArgmaxWERMetric(BaseMetric):
    def __init__(self, text_encoder: BaseTextEncoder, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.text_encoder = text_encoder

    def __call__(
        self, log_probs: Tensor, log_probs_length: Tensor, text: List[str], **kwargs
    ):
        wers = []
        predictions = torch.argmax(log_probs.cpu(), dim=-1).numpy()
        lengths = log_probs_length.detach().numpy()
        for log_prob_vec, length, target_text in zip(predictions, lengths, text):
            target_text = BaseTextEncoder.normalize_text(target_text)
            if hasattr(self.text_encoder, "ctc_decode"):
                pred_text = self.text_encoder.ctc_decode(log_prob_vec[:length])
            else:
                pred_text = self.text_encoder.decode(log_prob_vec[:length])
            wers.append(calc_wer(target_text, pred_text))
        return sum(wers) / len(wers)


class CustomBeamSearchWERMetric(BaseMetric):
    def __init__(self, text_encoder: BaseTextEncoder, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.text_encoder = text_encoder

    def __call__(
        self, log_probs: Tensor, log_probs_length: Tensor, text: List[str], **kwargs
    ):
        wers = []
        for log_probs, length, target_text in zip(log_probs, log_probs_length, text):
            target_text = BaseTextEncoder.normalize_text(target_text)
            pred_text = self.text_encoder.ctc_beam_search(
                log_probs.exp().cpu().numpy()[:length], beam_size=4
            )[0][0]
            wers.append(calc_wer(target_text, pred_text))
        return sum(wers) / len(wers)


class BeamSearchWERMetric(BaseMetric):
    def __init__(self, text_encoder: BaseTextEncoder, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.text_encoder = text_encoder

    def __call__(
        self, log_probs: Tensor, log_probs_length: Tensor, text: List[str], **kwargs
    ):
        wers = []
        for log_probs, length, target_text in zip(log_probs, log_probs_length, text):
            target_text = BaseTextEncoder.normalize_text(target_text)
            pred_text = self.text_encoder.pyctc_beam_search(
                log_probs.cpu().numpy(), length, beam_size=4
            )
            wers.append(calc_wer(target_text, pred_text))
        return sum(wers) / len(wers)


class LMBeamSearchWERMetric(BaseMetric):
    def __init__(self, text_encoder: BaseTextEncoder, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.text_encoder = text_encoder

    def __call__(
        self, log_probs: Tensor, log_probs_length: Tensor, text: List[str], **kwargs
    ):
        wers = []
        for log_probs, length, target_text in zip(log_probs, log_probs_length, text):
            target_text = BaseTextEncoder.normalize_text(target_text)
            pred_text = self.text_encoder.ptctc_beam_search_lm(
                log_probs.cpu().numpy(), length, beam_size=4
            )
            wers.append(calc_wer(target_text, pred_text))
        return sum(wers) / len(wers)
