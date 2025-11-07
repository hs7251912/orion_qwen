#    Copyright 2024 Orion Project
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

from transformers import Qwen2VLForConditionalGeneration
from transformers.modeling_outputs import CausalLMOutputWithPast

from .llava_arch import LlavaMetaForCausalLM


class OrionQwen2VlForCausalLM(Qwen2VLForConditionalGeneration, LlavaMetaForCausalLM):
    """
    Orion's adapter for Qwen2-VL, combining Qwen2-VL's architecture with
    LLaVA's multimodal input processing capability.
    
    This adapter enables seamless integration of Orion's BEV features with
    Qwen2-VL by reusing LLaVA's proven multimodal fusion mechanism.
    """
    
    def __init__(self, config, use_gen_token=False, use_critical_qa=False):
        # Initialize Qwen2-VL base model using proper super() call
        super().__init__(config)
        
        # Store Orion-specific configurations
        self.use_gen_token = use_gen_token
        self.hidden_size = config.hidden_size
        
        # Setup weighted mask for special tokens (same as LlavaLlamaForCausalLM)
        # These are number tokens: +-0.123456789
        number_tokens = [
            718, 448, 29900, 29889, 29896, 29906, 29941, 
            29946, 29945, 29953, 29955, 29947, 29929
        ]
        
        if use_gen_token:
            weighted_mask = torch.ones(self.config.vocab_size + 1)
            weighted_mask[number_tokens] = 1.0
        else:
            weighted_mask = torch.ones(self.config.vocab_size)
            weighted_mask[number_tokens] = 3.0
        
        if use_critical_qa:
            weighted_mask[number_tokens] = 3.0
            
        self.register_buffer("weighted_mask", weighted_mask)
        
        # Enable gradient checkpointing support
        self.supports_gradient_checkpointing = True
        self.gradient_checkpointing = False
    
    def get_model(self):
        """
        Required by LlavaMetaForCausalLM to access the base language model.
        For Qwen2-VL, the base model is stored in self.model.
        """
        return self.model
    
    def _set_gradient_checkpointing(self, module, value=False):
        """
        Enable or disable gradient checkpointing for the model.
        Required for compatibility with transformers' gradient_checkpointing_enable().
        """
        if isinstance(module, (type(self.model),)):
            module.gradient_checkpointing = value
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,  # BEV features from Orion
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
        return_ego_feature: Optional[bool] = False,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        """
        Forward pass that handles both text and vision inputs.
        
        The 'images' parameter receives Orion's vision_embeded (BEV features),
        which are then fused with text tokens using LLaVA's fusion mechanism.
        """
        
        # Step 1: Use LLaVA's fusion mechanism to merge BEV features with text
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
                images,  # This is vision_embeded from Orion (B, N_queries, C)
                image_sizes
            )
        else:
            new_input_ids = None
        
        # Step 2: Call Qwen2-VL's base model with the fused inputs_embeds
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # Call the parent's forward, which internally uses self.model
        # Note: Qwen2VLForConditionalGeneration has 'model' attribute after __init__
        outputs = super(Qwen2VLForConditionalGeneration, self).forward(
            input_ids=input_ids,  # Will be None after fusion
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,  # Fused text + vision embeddings
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        hidden_states = outputs[0]
        
        # Step 3: Extract ego_feature if needed (Orion-specific for trajectory planning)
        if return_ego_feature and new_input_ids is not None:
            if not isinstance(self.config.waypoint_token_idx, list):
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
        
        # Step 4: Compute logits and loss
        logits = self.lm_head(hidden_states)
        logits = logits.float()
        
        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Use weighted cross entropy loss
            loss_fct = CrossEntropyLoss(weight=self.weighted_mask.float())
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)
            loss = torch.nan_to_num(loss)
        
        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output
        
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
        """
        Generate method for text generation with multimodal inputs.
        """
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")
        
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
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                images,
                image_sizes=image_sizes
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)
        
        return super().generate(
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
        """
        Orion-specific inference method for extracting ego features.
        This is used for trajectory planning in autonomous driving.
        """
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")
        
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
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                images,
                image_sizes=image_sizes
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)
        
        output_attentions = self.config.output_attentions
        output_hidden_states = self.config.output_hidden_states
        return_dict = self.config.use_return_dict
        
        # Get model outputs
        outputs = self.model(
            input_ids=inputs,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=inputs_embeds,
            use_cache=True,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        hidden_states = outputs[0]
        
        if return_ego_feature:
            if not isinstance(self.config.waypoint_token_idx, list):
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
            raise ValueError("return_ego_feature must be True for inference_ego")
    
    def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                      inputs_embeds=None, **kwargs):
        """
        Prepare inputs for generation, including multimodal inputs.
        """
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            inputs['images'] = images
        if image_sizes is not None:
            inputs['image_sizes'] = image_sizes
        return inputs

