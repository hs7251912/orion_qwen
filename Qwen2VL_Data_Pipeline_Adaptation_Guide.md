# Dev Guide: Adapting the Data Pipeline for Qwen2-VL

This document provides a step-by-step guide for developers to adapt the multimodal data processing pipeline for the `Qwen2-VL-7B-Instruct` model.

## 1. Overview of Required Changes

The adaptation involves three main tasks:
1.  **Updating Configuration**: Pointing the data loader to the new Qwen2-VL tokenizer.
2.  **Adjusting Padding Logic**: Ensuring the correct padding token is used for the new model.
3.  **Implementing the Chat Template**: Modifying the conversation formatting logic to match Qwen2-VL's specific requirements.

## 2. Task 1: Update Dataset Configuration

*   **Effort**: Low (~0.5 hours)
*   **File(s) to Modify**: Your dataset configuration `.py` file(s) (e.g., `configs/_base_/datasets/nus-3d.py`).
*   **Action**:
    1.  Locate the `train_pipeline` and `test_pipeline` definitions within your dataset configuration.
    2.  Find the dictionary that defines the `LoadAnnoatationVQA` transform.
    3.  Change the `tokenizer` argument to the file path of your `Qwen2-VL-7B-Instruct` model checkpoint.

    **Example Change**:
    ```python
    # In your dataset config file
    train_pipeline = [
        # ... other transforms
        dict(
            type='LoadAnnoatationVQA',
            tokenizer='path/to/your/Qwen2-VL-7B-Instruct', # <-- UPDATE THIS PATH
            max_length=2048,
            # ... other args
        ),
        # ... other transforms
    ]
    ```

## 3. Task 2: Adjust Padding Token Logic

*   **Effort**: Low (~0.5 hours)
*   **File to Modify**: `mmcv/datasets/pipelines/transforms_3d.py`
*   **Action**:
    1.  Open the specified file and navigate to the `LoadAnnoatationVQA.__init__` method.
    2.  Replace the line `self.tokenizer.pad_token = self.tokenizer.unk_token`.
    3.  This change ensures that the tokenizer uses the correct end-of-sentence token for padding, which is standard practice for Qwen2 models.

    **Code Snippet**:
    ```python:mmcv/datasets/pipelines/transforms_3d.py
    # In class LoadAnnoatationVQA:
    #   def __init__(...):
            self.tokenizer =  AutoTokenizer.from_pretrained(...)
            # ...
            # self.tokenizer.pad_token = self.tokenizer.unk_token # <-- DELETE THIS LINE
            if self.tokenizer.pad_token is None: # <-- ADD THESE LINES
                self.tokenizer.pad_token = self.tokenizer.eos_token
            # ...
    ```

## 4. Task 3: Implement Qwen2-VL Chat Template

*   **Effort**: Medium (~2-4 hours)
*   **File to Modify**: `mmcv/datasets/data_utils/data_utils.py`
*   **Action**: This is the most critical step. You will create a new function to handle Qwen2-VL's chat format and hook it into the main `preprocess` function.

    **Step 4.1: Create a New `preprocess_qwen2_vl` Function**
    Add a new function to the file that takes the conversation data (`sources`) and tokenizer as input and formats it according to the Qwen2-VL template.

    **Conceptual Implementation**:
    ```python
    # In mmcv/datasets/data_utils/data_utils.py

    def preprocess_qwen2_vl(sources, tokenizer, has_image=False):
        conversations = []
        roles = {"human": "user", "gpt": "assistant"}

        for i, source in enumerate(sources):
            # Qwen2-VL template uses special tokens: <|im_start|> and <|im_end|>
            system_prompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            
            conversation = system_prompt
            for j, sentence in enumerate(source):
                role = roles[sentence["from"]]
                message = sentence["value"]
                # For the first user message, prepend the image token if it exists
                if j == 0 and has_image:
                    message = DEFAULT_IMAGE_TOKEN + '\n' + message
                
                conversation += f"<|im_start|>{role}\n{message}<|im_end|>\n"
            
            # The final turn must be from the assistant
            conversation += "<|im_start|>assistant\n"
            conversations.append(conversation)

        # Tokenize the formatted conversations
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

        targets = input_ids.clone()

        # Mask user prompts in labels. The logic here needs to be precise.
        # It should find the start of each assistant message and mask everything before it.
        # This is a simplified example; the actual implementation requires careful
        # handling of token indices.
        for i in range(len(conversations)):
            # This is a placeholder for the actual masking logic, which is complex.
            # You need to find the token indices for "<|im_start|>assistant\n" 
            # and mask all tokens before it for each turn.
            pass # Implement masking logic here

        return dict(input_ids=input_ids, labels=targets)
    ```

    **Step 4.2: Integrate the New Function**
    Modify the main `preprocess` function to call your new function. You'll need a way to specify which conversation format to use. A common way is to check the tokenizer name or add a new conversation mode.

    **Example Integration**:
    ```python
    # In mmcv/datasets/data_utils/data_utils.py

    def preprocess(sources, tokenizer, has_image=False, ...):
        # Add a new condition to check for Qwen2-VL
        if 'qwen2' in tokenizer.name_or_path.lower():
            return preprocess_qwen2_vl(sources, tokenizer, has_image=has_image)

        if conversation_lib.default_conversation.sep_style == ...:
            # ... existing logic
    ```
