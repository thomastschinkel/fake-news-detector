"""Custom FakeNewsDetector model definition."""

import torch
from torch import nn
from transformers import PreTrainedModel, RobertaConfig, RobertaModel
from transformers.modeling_outputs import SequenceClassifierOutput


class FakeNewsConfig(RobertaConfig):
    model_type = "fake_news_detector"

    def __init__(
        self,
        classifier_hidden_1: int = 512,
        classifier_hidden_2: int = 128,
        classifier_dropout_1: float = 0.3,
        classifier_dropout_2: float = 0.2,
        legacy_logits_output: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.classifier_hidden_1 = classifier_hidden_1
        self.classifier_hidden_2 = classifier_hidden_2
        self.classifier_dropout_1 = classifier_dropout_1
        self.classifier_dropout_2 = classifier_dropout_2
        self.legacy_logits_output = legacy_logits_output


class FakeNewsDetector(PreTrainedModel):
    config_class = FakeNewsConfig
    base_model_prefix = "roberta"

    def __init__(self, config: FakeNewsConfig):
        super().__init__(config)

        self.num_labels = config.num_labels
        self.roberta = RobertaModel(config, add_pooling_layer=False)

        concat_size = config.hidden_size * 2
        self.classifier = nn.Sequential(
            nn.Linear(concat_size, config.classifier_hidden_1),
            nn.LayerNorm(config.classifier_hidden_1),
            nn.GELU(),
            nn.Dropout(config.classifier_dropout_1),
            nn.Linear(config.classifier_hidden_1, config.classifier_hidden_2),
            nn.GELU(),
            nn.Dropout(config.classifier_dropout_2),
            nn.Linear(config.classifier_hidden_2, config.num_labels),
        )
        self.loss_fn = nn.CrossEntropyLoss()

        self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        return_dict=None,
        **kwargs,
    ):
        if input_ids is None:
            raise ValueError("input_ids must be provided")

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        return_dict = (
            return_dict
            if return_dict is not None
            else self.config.use_return_dict
        )

        out = self.roberta(
            input_ids=input_ids, attention_mask=attention_mask
        )
        hidden = out.last_hidden_state

        cls_token = hidden[:, 0, :]
        mask_expanded = attention_mask.unsqueeze(-1).float()
        mean_pooled = (hidden * mask_expanded).sum(1)
        mean_pooled = mean_pooled / mask_expanded.sum(1).clamp(min=1e-9)

        combined = torch.cat([cls_token, mean_pooled], dim=1)
        logits = self.classifier(combined)

        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels)

        if not return_dict:
            output = (logits,) + out[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=out.hidden_states,
            attentions=out.attentions,
        )