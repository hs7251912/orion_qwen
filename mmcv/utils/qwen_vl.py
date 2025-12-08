from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_outputs import CausalLMOutputWithPast

# Reuse constants aligned with llava_arch expectations
IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = -200


def add_special_token(special_token_list, tokenizer, model):
    num_new_tokens = tokenizer.add_tokens(special_token_list, special_tokens=True)
    model.resize_token_embeddings(len(tokenizer))
    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


class QwenVLMForCausalLM(nn.Module):
    """
    Thin wrapper over HF Qwen2.5-VL (or compatible Qwen-VL) that mirrors the
    minimal LLaVA interface used by Orion:
      - prepare_inputs_labels_for_multimodal: splice vision tokens at IMAGE_TOKEN_INDEX
      - forward / generate: accept images as precomputed tokens (B, N, hidden)
      - inference_ego: return hidden state at waypoint token positions
    """

    def __init__(self, hf_model: AutoModelForCausalLM):
        super().__init__()
        self.model = hf_model
        self.config = self.model.config
        self.hidden_size = getattr(self.config, 'hidden_size', self.model.get_input_embeddings().embedding_dim)
        self.pretraining_tp = getattr(self.config, 'pretraining_tp', 1)
        self.lm_head = self.model.get_output_embeddings()
        # compatibility flags
        self.use_gen_token = False
        self.register_buffer("weighted_mask", torch.ones(self.config.vocab_size), persistent=False)

    @classmethod
    def from_pretrained(cls, base_model: str, torch_dtype=torch.float32, device_map='cpu', **kwargs):
        # trust_remote_code is typically required for Qwen
        hf_model = AutoModelForCausalLM.from_pretrained(
            base_model, torch_dtype=torch_dtype, device_map=device_map, trust_remote_code=True, **kwargs
        )
        return cls(hf_model)

    def get_model(self):
        return self.model

    def prepare_inputs_labels_for_multimodal(
        self,
        input_ids,
        position_ids,
        attention_mask,
        past_key_values,
        labels,
        image_features,
        image_sizes,
    ):
        if image_features is None or input_ids.shape[1] == 1:
            return input_ids, position_ids, attention_mask, past_key_values, None, labels, None

        if isinstance(image_features, list):
            temp_image_features = []
            for b_id in range(len(image_features[0])):
                for img_id in range(len(image_features)):
                    temp_image_features.append(image_features[img_id][b_id])
            image_features = temp_image_features
        else:
            image_features = image_features.reshape(image_features.shape[0], -1, self.hidden_size).to(dtype=self.model.dtype)

        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        input_ids = [cur_input_ids[cur_attention_mask.cpu()] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        new_input_embeds = []
        new_labels = []
        new_input_ids = []
        cur_image_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            if num_images == 0:
                cur_image_features = image_features[cur_image_idx]
                cur_input_embeds_1 = self.get_model().get_input_embeddings()(cur_input_ids)
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue

            image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []
            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[i]+1:image_token_indices[i+1]])
                cur_labels_noim.append(cur_labels[image_token_indices[i]+1:image_token_indices[i+1]])
            split_sizes = [x.shape[0] for x in cur_labels_noim]
            cur_input_embeds = self.get_model().get_input_embeddings()(torch.cat(cur_input_ids_noim).to(image_features.device))
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)

            cur_new_input_embeds = []
            cur_new_labels = []
            cur_new_input_ids = []
            for i in range(num_images + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])
                cur_new_input_ids.append(cur_input_ids_noim[i])
                if i < num_images:
                    cur_image_features = image_features[cur_image_idx]
                    cur_image_idx += 1
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))
                    cur_new_input_ids.append(torch.full((cur_image_features.shape[0],), IMAGE_TOKEN_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))
            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)
            cur_new_input_ids = torch.cat(cur_new_input_ids)
            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)
            new_input_ids.append(cur_new_input_ids)

        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        new_inputs_ids_padded = torch.zeros((batch_size, max_len), dtype=new_input_ids[0].dtype, device=new_input_ids[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

        for i, (cur_new_embed, cur_new_labels, cur_new_input_ids) in enumerate(zip(new_input_embeds, new_labels, new_input_ids)):
            cur_len = cur_new_embed.shape[0]
            new_input_embeds_padded.append(torch.cat((
                cur_new_embed,
                torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)
            ), dim=0))
            if cur_len > 0:
                new_labels_padded[i, :cur_len] = cur_new_labels
                new_inputs_ids_padded[i, :cur_len] = cur_new_input_ids
                attention_mask[i, :cur_len] = True
                position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None

        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels, new_inputs_ids_padded

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
        return_ego_feature: Optional[bool] = False,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels,
                new_input_ids
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                image_sizes
            )
        else:
            new_input_ids = None

        outputs = self.model(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        hidden_states = outputs[0]

        if return_ego_feature:
            if not isinstance(getattr(self.config, 'waypoint_token_idx', -1), list):
                loc_positions = (new_input_ids == self.config.waypoint_token_idx)
                selected_hidden_states = hidden_states[loc_positions.to(device=hidden_states.device)]
            else:
                loc_positions_list = []
                for new_id in new_input_ids:
                    loc_positions = torch.zeros_like(new_id).to(torch.bool)
                    for token_id in self.config.waypoint_token_idx:
                        if token_id in new_id:
                            loc_positions = torch.logical_or(loc_positions, new_id == token_id)
                    loc_positions_list.append(loc_positions)
                loc_positions = torch.stack(loc_positions_list, dim=0)
                selected_hidden_states = hidden_states[loc_positions.to(device=hidden_states.device)]

        if self.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.config.vocab_size // self.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss(weight=self.weighted_mask.float())
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)
            loss = torch.nan_to_num(loss)

        if return_ego_feature:
            return CausalLMOutputWithPast(
                loss=loss,
                logits=logits,
                past_key_values=outputs.past_key_values,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            ), selected_hidden_states
        else:
            return CausalLMOutputWithPast(
                loss=loss,
                logits=logits,
                past_key_values=outputs.past_key_values,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if images is not None:
            (
                inputs,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _,
                _
            ) = self.prepare_inputs_labels_for_multimodal(
                inputs, position_ids, attention_mask, None, None, images, image_sizes=image_sizes
            )
        else:
            inputs_embeds = self.get_model().get_input_embeddings()(inputs)

        return self.model.generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs
        )

    @torch.no_grad()
    def inference_ego(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        return_ego_feature=False,
        **kwargs,
    ):
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if images is not None:
            (
                inputs,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _,
                new_input_ids
            ) = self.prepare_inputs_labels_for_multimodal(
                inputs, position_ids, attention_mask, None, None, images, image_sizes=image_sizes
            )
        else:
            inputs_embeds = self.get_model().get_input_embeddings()(inputs)

        outputs = self.model(
            input_ids=inputs,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=inputs_embeds,
            use_cache=True,
            output_attentions=self.config.output_attentions if hasattr(self.config, 'output_attentions') else False,
            output_hidden_states=self.config.output_hidden_states if hasattr(self.config, 'output_hidden_states') else False,
            return_dict=True,
        )
        hidden_states = outputs[0]
        if return_ego_feature:
            if not isinstance(getattr(self.config, 'waypoint_token_idx', -1), list):
                loc_positions = (new_input_ids == self.config.waypoint_token_idx)
                selected_hidden_states = hidden_states[loc_positions.to(device=hidden_states.device)]
            else:
                loc_positions_list = []
                for new_id in new_input_ids:
                    loc_positions = torch.zeros_like(new_id).to(torch.bool)
                    for token_id in self.config.waypoint_token_idx:
                        if token_id in new_id:
                            loc_positions = torch.logical_or(loc_positions, new_id == token_id)
                    loc_positions_list.append(loc_positions)
                loc_positions = torch.stack(loc_positions_list, dim=0)
                selected_hidden_states = hidden_states[loc_positions.to(device=hidden_states.device)]
            return selected_hidden_states
        else:
            assert False

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs = self.model.prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            inputs['images'] = images
        if image_sizes is not None:
            inputs['image_sizes'] = image_sizes
        return inputs


