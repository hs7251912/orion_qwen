# Design Proposal: Replacing LLaVA-LLaMA with Qwen2-VL-7B-Instruct

**Version**: 1.0  
**Date**: 2025-11-03

---

## 1. Executive Summary

This document outlines the technical design and implementation plan for replacing the `LlavaLlamaForCausalLM` model with `Qwen2-VL-7B-Instruct` within the Orion VLM fusion architecture.

The primary goal is to leverage the advanced capabilities of Qwen2-VL while maintaining the existing end-to-end driving task pipeline. The core technical challenge revolves around adapting the architecture to the dimensional differences between the two models: LLaMA's hidden size is **4096**, whereas Qwen2-7B's is **3584**.

This migration requires targeted modifications in three key areas:
1.  **Vision Feature Projection**: Adjusting the output dimension of the projection layers that process visual and vehicle state queries.
2.  **VLM Core Integration**: Replacing the model loading and input preparation logic to be compatible with Qwen2-VL's specific architecture and tokenization scheme.
3.  **Downstream Task Head**: Updating the trajectory decoder to accept the 3584-dimension ego feature vector produced by the new VLM.

---

## 2. Architectural Analysis & Core Changes

### 2.1. Motivation

Migrating to Qwen2-VL-7B-Instruct offers several potential advantages:
*   **Improved Performance**: Access to a more recent and powerful VLM.
*   **Better Multilingual & Code Support**: Enhanced capabilities for complex, instruction-based reasoning.
*   **Active Development**: Alignment with a model that is under active development and support.

### 2.2. Key Dimensional Mismatch

The central challenge is the incompatibility of hidden state dimensions between the two models.

| Component             | LLaVA-LLaMA (Current) | Qwen2-VL (Proposed) | Action Required                               |
| --------------------- | --------------------- | ------------------- | --------------------------------------------- |
| **VLM Hidden Size**   | `4096`                | `3584`              | **Core constraint**                           |
| Vision Feature Output | `4096`                | `3584`              | Modify projection layers                      |
| Ego Feature Vector    | `4096`                | `3584`              | Output dimension changes automatically        |
| Trajectory Decoder    | `4096` (Input Dim)    | `3584` (Input Dim)  | Modify decoder's input layer                  |

### 2.3. Required Architectural Modifications

1.  **Vision Feature Projector**:
    The linear layers that project the Scene Queries, History Queries, and CAN Bus features into the VLM's embedding space must be updated. The `out_features` parameter of these layers needs to be changed from 4096 to 3584.

2.  **VLM Integration**:
    The current model loading mechanism for `LlavaLlamaForCausalLM` will be replaced with `Qwen2_5_VLForConditionalGeneration`. This includes pointing to the new local model path and ensuring the correct `torch_dtype` and `device_map` are used.

3.  **Input Embedding Preparation**:
    Qwen2-VL has a specific format for multimodal inputs. The vision tokens must be embedded and spliced into the text token sequence, potentially using special tokens like `<img>` and `</img>`. The `prepare_inputs_labels_for_multimodal` function will require a significant rewrite to accommodate this new format.

4.  **Ego Feature Extraction**:
    The core logic of extracting the hidden state from a specific token position (e.g., `<ego_wp>`) remains valid. However, the resulting `ego_feature` tensor will now have a shape of `(batch_size, 3584)`.

5.  **Trajectory Decoder**:
    The downstream trajectory prediction head, which currently accepts a 4096-dimensional vector, must be modified. Its initial linear layer must be changed to accept an `in_features` of 3584.

---

## 3. File-by-File Implementation Guide

### 3.1. `mmcv/models/dense_heads/orion_head.py`

*   **Change**: Modify the linear projection layers for visual and CAN bus features.
*   **Location**: Inside the `OrionHead` class constructor (`__init__`).
*   **Details**:
    *   Locate the `nn.Linear` layers responsible for projecting `det_query`, `map_query`, and `can_bus_query`.
    *   Change their `out_features` from `4096` to `3584`.

    ```python
    # Example Snippet (Conceptual)
    # In OrionHead.__init__
    self.det_proj = nn.Linear(self.embed_dims, 3584)
    self.map_proj = nn.Linear(self.embed_dims, 3584)
    self.can_bus_proj = nn.Linear(can_bus_input_dim, 3584)
    ```

### 3.2. `mmcv/models/detectors/orion.py`

*   **Change**: Replace the VLM model class and path.
*   **Location**: Inside the `Orion` class constructor (`__init__`).
*   **Details**:
    *   Import `Qwen2_5_VLForConditionalGeneration` from `transformers`.
    *   Update the `from_pretrained` call to use this new class.
    *   Change the `model_path` variable to point to `'/root/autodl-tmp/Orion_modify/ckpts/Qwen2-VL-7B-Instruct'`.

    ```python
    # Example Snippet (Conceptual)
    # In Orion.__init__
    from transformers import Qwen2_5_VLForConditionalGeneration as QwenVLModel

    self.llm = QwenVLModel.from_pretrained(
        '/root/autodl-tmp/Orion_modify/ckpts/Qwen2-VL-7B-Instruct',
        torch_dtype=torch.bfloat16,
        # ... other args
    )
    ```

### 3.3. `mmcv/utils/llava_arch.py` (or equivalent)

*   **Change**: Re-implement the multimodal input preparation logic.
*   **Location**: The `prepare_inputs_labels_for_multimodal` function.
*   **Details**:
    *   The current implementation splices vision embeddings directly into the text embedding sequence.
    *   The new implementation must be aligned with how Qwen2-VL processes vision inputs. This typically involves using the processor to create image-specific tokens and then replacing placeholder tokens in the text with the projected vision embeddings. The exact implementation will depend on the specifics of the `Qwen2_5_VLProcessor`.

### 3.4. Trajectory Head (e.g., in `mmcv/models/dense_heads/orion_head.py`)

*   **Change**: Update the input dimension of the trajectory decoder.
*   **Location**: The module responsible for trajectory prediction (likely part of `OrionHead`).
*   **Details**:
    *   Identify the `nn.Module` that takes the `ego_feature` as input.
    *   Modify its first `nn.Linear` layer's `in_features` from `4096` to `3584`.

    ```python
    # Example Snippet (Conceptual)
    # In the Trajectory Decoder module's __init__
    self.decoder_input_layer = nn.Linear(3584, self.decoder_hidden_dim)
    ```

---

## 4. New Data Flow Diagram (ASCII)

This diagram illustrates the updated data flow with the new dimensions.

```
Inputs
|
+-- [Images] -> ResNet -> FPN -> BEVEncoder -> (256, 256) Feature Maps
|
+-- [CAN Bus] -> (89)
|
+-- [Text Prompt] -> Tokenizer -> (seq_len)

      |
      |____________________________________________________________________
      |                                                                   |
(256, 256) Feature Maps                                               [CAN Bus] (89)
      |                                                                   |
      +-- Det/Map Heads -> [Scene Queries (256, 256)], [History Queries (256, 256)]
      |                                                                   |
      |------------------------------+------------------------------------|
      |                              |                                    |
[Scene Queries] (256, 256)     [History Queries] (256, 256)         [CAN Bus] (89)
      |                              |                                    |
    nn.Linear(256, 3584)         nn.Linear(256, 3584)                nn.Linear(89, 3584)
      |                              |                                    |
[Det Tokens] (256, 3584)       [Map Tokens] (256, 3584)            [CAN Token] (1, 3584)
      |                              |                                    |
      |___________ Concatenate ________|__________________________________|
                          |
            [Vision Tokens] (513, 3584)
                          |
                          |
+-------------------------+--------------------------+
|                                                    |
[Text Embeddings] (seq_len, 3584)      [Vision Tokens] (513, 3584)
|                                                    |
+--- `prepare_inputs_for_multimodal` (Splice vision into text)
                          |
            [Combined Embeddings] (batch_size, new_seq_len, 3584)
                          |
+-------------------------V--------------------------+
|            Qwen2-VL-7B-Instruct Model              |
|                                                    |
|   Outputs: Hidden States (batch, seq_len, 3584)    |
+----------------------------------------------------+
                          |
+--- Extract hidden state at `<ego_wp>` token position
                          |
              [Ego Feature] (batch_size, 3584)
                          |
+-------------------------V--------------------------+
|             Trajectory Decoder (Input Dim: 3584)   |
+----------------------------------------------------+
                          |
              [Predicted Trajectory] (batch_size, 6, 2)

```
