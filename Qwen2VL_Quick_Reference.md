# Quick Reference: VLM Fusion with Qwen2-VL-7B-Instruct

**Version**: 1.0 (Qwen2-VL Adaptation)  
**Date**: 2025-11-03

---

## 1. Core Architecture Flow (ASCII)

This diagram outlines the data flow with the Qwen2-VL model and the updated **3584** dimensions.

```
Inputs
|
+-- [Images] -> BEVEncoder -> [Scene Queries (256, 256)], [History Queries (256, 256)]
|
+-- [CAN Bus] -> (89)
|
+-- [Text Prompt] -> Tokenizer -> (seq_len)

      |
      |____________________________________________________________________
      |                                                                   |
[Scene/History Queries]                                              [CAN Bus] (89)
      |                                                                   |
    nn.Linear(256, 3584)                                             nn.Linear(89, 3584)
      |                                                                   |
[Det/Map Tokens] (512, 3584)                                       [CAN Token] (1, 3584)
      |                                                                   |
      |_____________________ Concatenate _________________________________|
                                  |
                      [Vision Tokens] (513, 3584)
                                  |
+---------------------------------V----------------------------------+
|                    Qwen2-VL-7B-Instruct Model                      |
|                                                                    |
|   Input: Spliced Text & Vision Embeddings (batch, seq_len, 3584)   |
|   Output: Hidden States (batch, seq_len, 3584)                     |
+--------------------------------------------------------------------+
                                  |
+--------------- Extract hidden state at `<ego_wp>` token
                                  |
                    [Ego Feature] (batch_size, 3584)
                                  |
+---------------------------------V----------------------------------+
|           Trajectory Decoder (Input Dim: 3584)                     |
+--------------------------------------------------------------------+
                                  |
                [Predicted Trajectory] (batch_size, 6, 2)
```

---

## 2. Dimension Quick Reference Table

| Stage                         | Tensor Name               | Shape                                 | Notes                                   |
| ----------------------------- | ------------------------- | ------------------------------------- | --------------------------------------- |
| **Input Queries**             | `det_query`, `map_query`  | `(B, 256, 256)`                       | From BEV detection/map heads.           |
| **Input CAN Bus**             | `can_bus_input`           | `(B, 89)`                             | Raw vehicle state data.                 |
| **Projection**                | `output_projection`       | `nn.Linear(256, 3584)`                | Projects vision queries.                |
| **CAN Bus Embedding**         | `can_bus_embed`           | `nn.Sequential` -> `(B, 1, 3584)`     | Encodes CAN bus data.                   |
| **Vision Tokens**             | `vision_embeded`          | `(B, 513, 3584)`                      | Concatenated projected features.        |
| **VLM Hidden State**          | `hidden_states`           | `(B, seq_len, 3584)`                  | Output from Qwen2-VL model.             |
| **Ego Feature**               | `ego_feature`             | `(B, 3584)`                           | Extracted from a special token position.|
| **Trajectory Decoder Input**  | -                         | `(B, 3584)`                           | Input to the final planning head.       |
| **Final Trajectory**          | `predicted_trajectory`    | `(B, 6, 2)`                           | 6 waypoints for the next 3 seconds.     |

---

## 3. Vision Token Decomposition (513 Tokens)

The 513 vision tokens fed into the VLM are a concatenation of three sources, each projected to **3584** dimensions.

| Token Type        | Count | Original Dimension | Projected Dimension | Source                               |
| ----------------- | ----- | ------------------ | ------------------- | ------------------------------------ |
| Detection Queries | 256   | 256                | 3584                | `OrionHead` output for objects.      |
| Map Queries       | 256   | 256                | 3584                | `OrionHeadMap` output for lanes, etc.|
| CAN Bus Feature   | 1     | 89                 | 3584                | `can_bus_embed` output.              |
| **Total**         | **513** | -                  | **3584**            | Concatenated `vision_embeded` tensor.|

---

## 4. Key Code Snippets (Post-Migration)

### `mmcv/models/dense_heads/orion_head.py` - Projection Layers

```python
# In OrionHead.__init__
self.output_dims = 3584

# In OrionHead._init_layers
self.output_projection = nn.Linear(self.embed_dims, self.output_dims) # embed_dims is 256

self.can_bus_embed = nn.Sequential(
    nn.Linear(89, self.embed_dims*4),
    nn.ReLU(),
    nn.Linear(self.embed_dims*4, self.output_dims)
)
```

### `mmcv/models/detectors/orion.py` - Trajectory Decoder (MLP Example)

```python
# In Orion.__init__ where use_mlp_decoder=True
self.waypoint_decoder = nn.Sequential(
    nn.Linear(3584, 3584 // 2),
    nn.GELU(),
    nn.Linear(3584 // 2, 6*2),
)
```

### `mmcv/models/detectors/orion.py` - VLM Integration

```python
# In Orion.__init__
from transformers import Qwen2_5_VLForConditionalGeneration as VLMForCausalLM

self.lm_head = load_model(
    lm_head_cfg,
    # ... other args
)

# In Orion.forward_pts_train
vision_embeded = torch.cat([vision_embeded_obj, vision_embeded_map], dim=1) # (B, 513, 3584)
vlm_loss, ego_feature = self.lm_head(
    # ... inputs
    images=vision_embeded,
    return_ego_feature=True
)
```

---

## 5. Special Token Mechanism

-   **`<image>` (or `IMAGE_TOKEN_INDEX=-200`)**: A placeholder token in the text prompt's `input_ids`.
-   **`prepare_inputs_labels_for_multimodal`**: This function finds the placeholder's location, removes it, and splices the `(B, 513, 3584)` `vision_embeded` tensor into the text token embeddings at that location.
-   **`<ego_wp>` (EGO_WAYPOINT_TOKEN)**: A special token used in prompts for planning tasks. The model is trained to output the `ego_feature` at this token's position in the final hidden states. This feature is then passed to the trajectory decoder.
