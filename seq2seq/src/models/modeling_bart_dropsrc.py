from typing import *
from transformers.models.bart.modeling_bart import (
    BartForConditionalGeneration,
    BartConfig,
    BartModel,
    BartPretrainedModel,
    BartEncoder,
)
import torch
import torch.nn as nn
import math
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqLMOutput

# _expand_mask was renamed / removed in recent transformers releases (>4.41).
# Attempt to import it; if missing, define a minimal compatible fallback.
try:
    from transformers.models.bart.modeling_bart import _expand_mask  # type: ignore
except ImportError:
    import torch

    def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: int | None = None):
        """Expand 2-D mask to 4-D causal mask used by BART attention layers.

        Args:
            mask: (bsz, src_len) with 1 for tokens that are NOT masked.
            dtype: dtype of the output (usually hidden_states.dtype).
            tgt_len: optional target length, defaults to src_len.
        Returns:
            (bsz, 1, tgt_len, src_len) tensor with 0 for keep, -inf for masked positions.
        """
        bsz, src_len = mask.shape
        tgt_len = tgt_len if tgt_len is not None else src_len
        expanded_mask = mask[:, None, None, :].to(dtype)
        inverted_mask = (1.0 - expanded_mask) * torch.finfo(dtype).min
        return inverted_mask.expand(bsz, 1, tgt_len, src_len)


class BartEncoderwithDropoutSrc(BartEncoder):
    def __init__(self, config: BartConfig, embed_tokens, src_dropout):
        super().__init__(config, embed_tokens)
        self.src_dropout = src_dropout
        self.embed_scale = math.sqrt(config.d_model) if getattr(config, "scale_embedding", False) else 1.0
        print('===================  src_dropout: ', src_dropout)
        self.post_init()

    def SRC_dropout(self, embedding_tokens, drop_prob):
        """
        dropout source word embedding to prevent from overfitting
        """
        if drop_prob == 0:
            return embedding_tokens
        keep_prob = 1 - drop_prob
        mask = (torch.randn(embedding_tokens.size()[:-1]) < keep_prob).unsqueeze(-1)
        embedding_tokens *= mask.eq(1).to(embedding_tokens.device)
        return embedding_tokens * (1 / keep_prob)
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutput]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input = input_ids
            input_ids = input_ids.view(-1, input_ids.shape[-1])
        elif inputs_embeds is not None:
            input = inputs_embeds[:, :, -1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if inputs_embeds is None:
            # inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale
            inputs_embeds = self.embed_tokens(input_ids)
            if self.training:
                inputs_embeds = self.SRC_dropout(inputs_embeds, self.src_dropout)

        inputs_embeds = inputs_embeds * self.embed_scale
        
        # Use the standard HuggingFace implementation approach for position embeddings
        # This is a direct port from the original BartEncoder implementation
        if attention_mask is None:
            if input_ids is not None:
                attention_mask = torch.ones_like(input_ids)
            else:
                attention_mask = torch.ones(inputs_embeds.shape[:2], device=inputs_embeds.device)
            
        # Extract batch size and sequence length and create position IDs
        batch_size, seq_length = input_ids.shape if input_ids is not None else inputs_embeds.shape[:2]
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        past_key_values_length = 0
            
        # Create position IDs - this is what BART's LearnedPositionalEmbedding expects
        position_ids = torch.arange(
            past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
        )
        position_ids = position_ids.unsqueeze(0).expand(batch_size, seq_length)
            
        # Create position embeddings using the correct format
        embed_pos = self.embed_positions(position_ids)

        hidden_states = inputs_embeds + embed_pos
        hidden_states = self.layernorm_embedding(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        # expand attention_mask
        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            attention_mask = _expand_mask(attention_mask, inputs_embeds.dtype)

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        # check if head_mask has a correct number of layers specified if desired
        if head_mask is not None:
            if head_mask.size()[0] != (len(self.layers)):
                raise ValueError(
                    f"The head_mask should be specified for {len(self.layers)} layers, but it is for"
                    f" {head_mask.size()[0]}."
                )

        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            to_drop = False
            if self.training:
                dropout_probability = torch.rand([])
                if dropout_probability < self.layerdrop:  # skip the layer
                    to_drop = True

            if to_drop:
                layer_outputs = (None, None)
            else:
                if self.gradient_checkpointing and self.training:

                    def create_custom_forward(module):
                        def custom_forward(*inputs):
                            return module(*inputs, output_attentions)

                        return custom_forward

                    layer_outputs = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(encoder_layer),
                        hidden_states,
                        attention_mask,
                        (head_mask[idx] if head_mask is not None else None),
                    )
                else:
                    layer_outputs = encoder_layer(
                        hidden_states,
                        attention_mask,
                        layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                        output_attentions=output_attentions,
                    )

                hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions
        )

class BartModelwithDropoutSrc(BartModel):
    def __init__(self, config: BartConfig, src_dropout):
        super().__init__(config)
        self.encoder = BartEncoderwithDropoutSrc(config, self.shared, src_dropout=src_dropout)
        self.post_init()

class BartForConditionalGenerationwithDropoutSrc(BartForConditionalGeneration):
    def __init__(self, config: BartConfig, src_dropout=0.0):
        super().__init__(config)
        self.model = BartModelwithDropoutSrc(config, src_dropout=src_dropout)
        self.post_init()