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
        print("dropout")
        print(classifier_dropout)
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(5*config.hidden_size+3, config.num_labels)

        self.init_weights()

    def set_weights(self,weights):
        self.weights = weights

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            entity_mask_start=None,
            entity_mask_end=None,
            context_entities_input_ids=None,
            context_entities_attention_mask=None,
            context_entities_token_type_ids=None,
            mentions_entities_input_ids=None,
            mentions_entities_attention_mask=None,
            mentions_entities_token_type_ids=None,
            mentions_abstracts_input_ids=None,
            mentions_abstracts_attention_mask=None,
            mentions_abstracts_token_type_ids=None,
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

        outputs_context_abstracts = self.bert(
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

        outputs_context_entities = self.bert(
            context_entities_input_ids,
            attention_mask=context_entities_attention_mask,
            token_type_ids=context_entities_token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=return_dict,
        )

        outputs_mentions_entities = self.bert(
            mentions_entities_input_ids,
            attention_mask=mentions_entities_attention_mask,
            token_type_ids=mentions_entities_token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=return_dict,
        )

        outputs_mentions_abstracts = self.bert(
            mentions_abstracts_input_ids,
            attention_mask=mentions_abstracts_attention_mask,
            token_type_ids=mentions_abstracts_token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=return_dict,
        )

        #entity_hidden_state_1 = outputs[1]
        entity_start_state = outputs_context_abstracts.last_hidden_state[entity_mask_start,:]
        entity_end_state = outputs_context_abstracts.last_hidden_state[entity_mask_end, :]

        entity_state_context_entities = outputs_context_entities[1]
        entity_state_mentions_entities = outputs_mentions_entities[1]
        entity_state_mentions_abstracts = outputs_mentions_abstracts[1]

        pooled_output = torch.cat([entity_start_state, entity_end_state, entity_state_context_entities, entity_state_mentions_entities, entity_state_mentions_abstracts], dim=1)
        pooled_output = self.dropout(pooled_output)

        pooled_output = torch.cat([pooled_output, adds_prior, adds_redirect, adds_surname], dim=1)
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