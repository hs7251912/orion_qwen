#    Copyright 2023 Haotian Liu
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


CONTROLLER_HEART_BEAT_EXPIRATION = 30
WORKER_HEART_BEAT_INTERVAL = 15

LOGDIR = "."

# Model Constants
IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = -200
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"


from abc import ABC, abstractmethod

import torch
import torch.nn as nn

class LlavaMetaModel:

    def __init__(self, config):
        super(LlavaMetaModel, self).__init__(config)

class LlavaMetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self):
        pass

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()

    def prepare_inputs_labels_for_multimodal(
        self, input_ids, position_ids, attention_mask, past_key_values, labels, image_features, image_sizes
    ):
        vision_tower = self.get_vision_tower()
        if vision_tower is None or image_features is None or input_ids.shape[1] == 1:
            if past_key_values is not None and vision_tower is not None and image_features is not None and input_ids.shape[1] == 1:
                target_shape = past_key_values[-1][-1].shape[-2] + 1
                attention_mask = torch.cat((attention_mask, torch.ones(
                    (attention_mask.shape[0], target_shape - attention_mask.shape[1]),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device
                )), dim=1)
                position_ids = torch.sum(attention_mask, dim=1).unsqueeze(-1) - 1
            return input_ids, position_ids, attention_mask, past_key_values, None, labels

        # The image_features are pre-computed, so we just need to splice them in.
        # This implementation is a simplified version for Qwen2-VL based on the original LLaVA logic.
        # It assumes a single special token is used as a placeholder for all vision features.

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

        # Find the placeholder token for images, assuming it's IMAGE_TOKEN_INDEX
        image_token_indices = torch.where(input_ids == IMAGE_TOKEN_INDEX)

        new_input_embeds = []
        new_labels = []
        for batch_idx, cur_input_ids in enumerate(input_ids):
            image_token_idx = image_token_indices[1][image_token_indices[0] == batch_idx]
            
            if len(image_token_idx) == 0: # No image token in this sample
                new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids))
                new_labels.append(labels[batch_idx])
                continue

            # Assuming one image token placeholder per sample
            image_token_idx = image_token_idx[0]

            # Embed text tokens before and after the image placeholder
            embed_tokens = self.get_model().embed_tokens
            pre_image_embeds = embed_tokens(cur_input_ids[:image_token_idx])
            post_image_embeds = embed_tokens(cur_input_ids[image_token_idx+1:])

            # Splice in the image features
            cur_image_features = image_features[batch_idx].to(pre_image_embeds.dtype)
            
            cur_new_input_embeds = torch.cat([pre_image_embeds, cur_image_features, post_image_embeds], dim=0)
            new_input_embeds.append(cur_new_input_embeds)

            # Adjust labels accordingly
            pre_labels = labels[batch_idx, :image_token_idx]
            post_labels = labels[batch_idx, image_token_idx+1:]
            image_labels = torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=labels.device, dtype=labels.dtype)
            
            cur_new_labels = torch.cat([pre_labels, image_labels, post_labels], dim=0)
            new_labels.append(cur_new_labels)

        # Pad the batch to the maximum length
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        padded_embeds = torch.zeros(batch_size, max_len, new_input_embeds[0].shape[1], dtype=new_input_embeds[0].dtype, device=new_input_embeds[0].device)
        padded_labels = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        attention_mask = torch.zeros(batch_size, max_len, dtype=torch.bool, device=input_ids.device)

        for i, (embed, label) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = embed.shape[0]
            padded_embeds[i, :cur_len] = embed
            padded_labels[i, :cur_len] = label
            attention_mask[i, :cur_len] = True

        if _labels is None:
            padded_labels = None
            
        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        return None, None, attention_mask, past_key_values, padded_embeds, padded_labels
