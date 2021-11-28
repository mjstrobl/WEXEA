from transformers import BertPreTrainedModel,BertModel
from torch import nn
import torch
from torch.nn import CrossEntropyLoss, MSELoss, BCEWithLogitsLoss
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPoolingAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    MaskedLMOutput,
    MultipleChoiceModelOutput,
    NextSentencePredictorOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)


class BertForEntityClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.bert = BertModel(config)

        classifier_dropout = config.hidden_dropout_prob
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size+3, config.num_labels)

        self.init_weights()

    def set_weights(self,weights):
        self.weights = weights

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            entity_mask=None,
            adds_prior=None,
            adds_redirect=None,
            adds_surname=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=return_dict,
        )

        #last_hidden_state = outputs[1]
        #pooled_output = torch.cat([last_hidden_state, adds_prior, adds_redirect, adds_surname], dim=1)
        #pooled_output = self.dropout(result)
        #logits = self.classifier(pooled_output)

        entity_hidden_state_1 = torch.mul(outputs.last_hidden_state,entity_mask)
        entity_hidden_state_2 = torch.sum(entity_hidden_state_1,dim=1)
        pooled_output = torch.cat([entity_hidden_state_2, adds_prior, adds_redirect, adds_surname], dim=1)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        '''if not return_dict:
            output = (logits,) + outputs_context[2:]
            return ((loss,) + output) if loss is not None else output'''

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits
        )