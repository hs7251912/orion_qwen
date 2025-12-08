import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Define special tokens used in the project
EGO_WAYPOINT_TOKEN = "<ego_wp>"

def add_special_token(special_token_list, tokenizer, model):
    """
    Adds special tokens to the tokenizer and model, and initializes the new token
    embeddings with the average of the existing embeddings.
    """
    num_new_tokens = tokenizer.add_tokens(special_token_list, special_tokens=True)
    model.resize_token_embeddings(len(tokenizer))
    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg

def load_model(lm_head_config, use_lora, frozen, lm_kwargs, fp16_infer):
    """
    Dynamically loads a Vision Language Model and its tokenizer based on the provided configuration.

    This function handles:
    - Loading different VLM architectures (LLaVA, InternVL, etc.).
    - Applying LoRA weights.
    - Freezing model parameters for feature extraction.
    - Handling floating point precision (FP16).
    - Adding custom special tokens required for tasks like trajectory generation.
    """
    model_path = lm_head_config['model_path']
    model_name = lm_head_config.get('model_name', 'llava')
    tokenizer_path = lm_head_config.get('tokenizer_path', model_path)

    # Determine the torch dtype for model loading
    torch_dtype = torch.float16 if fp16_infer else torch.float32

    # Load the appropriate model class based on model_name
    if model_name == 'llava':
        from .llava_llama import LlavaLlamaForCausalLM
        model_class = LlavaLlamaForCausalLM
    elif model_name == 'internvl':
        from .internvl_model import InternVLForCausalLM
        model_class = InternVLForCausalLM
    else:
        model_class = AutoModelForCausalLM

    model = model_class.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
        **lm_kwargs
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

    # Add special tokens for trajectory generation if specified
    if lm_kwargs.get('use_gen_token', False):
        add_special_token([EGO_WAYPOINT_TOKEN], tokenizer, model)
        # Store the waypoint token ID in the model's config for easy access
        waypoint_token_id = tokenizer(EGO_WAYPOINT_TOKEN, add_special_tokens=False).input_ids[0]
        model.config.waypoint_token_idx = waypoint_token_id
    
    model.model_name = model_name

    # Apply and merge LoRA weights if specified
    if use_lora:
        model = PeftModel.from_pretrained(model, model_path)
        model = model.merge_and_unload()

    # Freeze model parameters if specified
    if frozen:
        for param in model.parameters():
            param.requires_grad = False

    return model, tokenizer
