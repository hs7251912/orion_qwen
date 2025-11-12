import math
import numpy as np
from nuscenes.eval.common.utils import quaternion_yaw, Quaternion
from nuscenes.utils.data_classes import Box as NuScenesBox
import pyquaternion
from mmcv.datasets.data_utils.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN

from . import conversation as conversation_lib
import transformers
import torch
from typing import Dict, Optional, Sequence, List
import copy
import os

def output_to_nusc_box(detection):
    """Convert the output to the box class in the nuScenes.
    Args:
        detection (dict): Detection results.
            - boxes_3d (:obj:`BaseInstance3DBoxes`): Detection bbox.
            - scores_3d (torch.Tensor): Detection scores.
            - labels_3d (torch.Tensor): Predicted box labels.
    Returns:
        list[:obj:`NuScenesBox`]: List of standard NuScenesBoxes.
    """
    box3d = detection['boxes_3d']
    scores = detection['scores_3d'].numpy()
    labels = detection['labels_3d'].numpy()
    if 'track_ids' in detection:
        ids = detection['track_ids'].numpy()
    else:
        ids = np.ones_like(labels)

    box_gravity_center = box3d.gravity_center.numpy()
    box_dims = box3d.dims.numpy()
    box_yaw = box3d.yaw.numpy()
    # TODO: check whether this is necessary
    # with dir_offset & dir_limit in the head
    box_yaw = -box_yaw - np.pi / 2

    box_list = []
    for i in range(len(box3d)):
        quat = pyquaternion.Quaternion(axis=[0, 0, 1], radians=box_yaw[i])
        velocity = (*box3d.tensor[i, 7:9], 0.0)
        # velo_val = np.linalg.norm(box3d[i, 7:9])
        # velo_ori = box3d[i, 6]
        # velocity = (
        # velo_val * np.cos(velo_ori), velo_val * np.sin(velo_ori), 0.0)
        box = NuScenesBox(
            box_gravity_center[i],
            box_dims[i],
            quat,
            label=labels[i],
            score=scores[i],
            velocity=velocity)
        box.token = ids[i]
        box_list.append(box)
    return box_list


def output_to_nusc_box_det(detection):
    """Convert the output to the box class in the nuScenes.

    Args:
        detection (dict): Detection results.

            - boxes_3d (:obj:`BaseInstance3DBoxes`): Detection bbox.
            - scores_3d (torch.Tensor): Detection scores.
            - labels_3d (torch.Tensor): Predicted box labels.

    Returns:
        list[:obj:`NuScenesBox`]: List of standard NuScenesBoxes.
    """
    if 'boxes_3d_det' in detection:
        box3d = detection['boxes_3d_det']
        scores = detection['scores_3d_det'].numpy()
        labels = detection['labels_3d_det'].numpy()
    else:
        box3d = detection['boxes_3d']
        scores = detection['scores_3d'].numpy()
        labels = detection['labels_3d'].numpy()

    box_gravity_center = box3d.gravity_center.numpy()
    box_dims = box3d.dims.numpy()
    box_yaw = box3d.yaw.numpy()
    # TODO: check whether this is necessary
    # with dir_offset & dir_limit in the head
    box_yaw = -box_yaw - np.pi / 2

    box_list = []
    for i in range(len(box3d)):
        quat = Quaternion(axis=[0, 0, 1], radians=box_yaw[i])
        velocity = (*box3d.tensor[i, 7:9], 0.0)
        box = NuScenesBox(
            box_gravity_center[i],
            box_dims[i],
            quat,
            label=labels[i],
            score=scores[i],
            velocity=velocity)
        box_list.append(box)
    return box_list


def lidar_nusc_box_to_global(info,
                             boxes,
                             classes,
                             eval_configs,
                             eval_version='detection_cvpr_2019'):
    """Convert the box from ego to global coordinate.
    Args:
        info (dict): Info for a specific sample data, including the
            calibration information.
        boxes (list[:obj:`NuScenesBox`]): List of predicted NuScenesBoxes.
        classes (list[str]): Mapped classes in the evaluation.
        eval_configs (object): Evaluation configuration object.
        eval_version (str, optional): Evaluation version.
            Default: 'detection_cvpr_2019'
    Returns:
        list: List of standard NuScenesBoxes in the global
            coordinate.
    """
    box_list = []
    keep_idx = []
    for i, box in enumerate(boxes):
        # Move box to ego vehicle coord system
        box.rotate(Quaternion(info['lidar2ego_rotation']))
        box.translate(np.array(info['lidar2ego_translation']))
        # filter det in ego.
        cls_range_map = eval_configs.class_range
        radius = np.linalg.norm(box.center[:2], 2)
        det_range = cls_range_map[classes[box.label]]
        if radius > det_range:
            continue
        # Move box to global coord system
        box.rotate(Quaternion(info['ego2global_rotation']))
        box.translate(np.array(info['ego2global_translation']))
        box_list.append(box)
        keep_idx.append(i)
    return box_list, keep_idx


def obtain_map_info(nusc,
                    nusc_maps,
                    sample,
                    patch_size=(102.4, 102.4),
                    canvas_size=(256, 256),
                    layer_names=['lane_divider', 'road_divider'],
                    thickness=10):
    """
    Export 2d annotation from the info file and raw data.
    """
    l2e_r = sample['lidar2ego_rotation']
    l2e_t = sample['lidar2ego_translation']
    e2g_r = sample['ego2global_rotation']
    e2g_t = sample['ego2global_translation']
    l2e_r_mat = Quaternion(l2e_r).rotation_matrix
    e2g_r_mat = Quaternion(e2g_r).rotation_matrix

    scene = nusc.get('scene', sample['scene_token'])
    log = nusc.get('log', scene['log_token'])
    nusc_map = nusc_maps[log['location']]
    if layer_names is None:
        layer_names = nusc_map.non_geometric_layers

    l2g_r_mat = (l2e_r_mat.T @ e2g_r_mat.T).T
    l2g_t = l2e_t @ e2g_r_mat.T + e2g_t
    patch_box = (l2g_t[0], l2g_t[1], patch_size[0], patch_size[1])
    patch_angle = math.degrees(Quaternion(matrix=l2g_r_mat).yaw_pitch_roll[0])

    map_mask = nusc_map.get_map_mask(
        patch_box, patch_angle, layer_names, canvas_size=canvas_size)
    map_mask = map_mask[-2] | map_mask[-1]
    map_mask = map_mask[np.newaxis, :]
    map_mask = map_mask.transpose((2, 1, 0)).squeeze(2)  # (H, W, C)

    erode = nusc_map.get_map_mask(patch_box, patch_angle, [
                                  'drivable_area'], canvas_size=canvas_size)
    erode = erode.transpose((2, 1, 0)).squeeze(2)

    map_mask = np.concatenate([erode[None], map_mask[None]], axis=0)
    return map_mask

def tokenizer_image_token(prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX, return_tensors=None):
    """
    适配多种 tokenizer 的图像 token 处理
    支持: LLaVA (使用 -200) 和 Qwen2-VL (使用特殊 token 序列)
    
    Args:
        prompt: 包含 <image> 占位符的文本
        tokenizer: tokenizer 实例
        image_token_index: LLaVA 使用的图像 token ID (默认 -200)
        return_tensors: 返回格式 ('pt' 或 None)
    """
    # 环境开关：强制使用 LLaVA 的 -200 占位（用于兼容 Orion 旧路径）
    force_legacy = os.getenv("ORION_FORCE_LEGACY_IMAGE_TOKEN", "0").lower() in ("1", "true", "yes", "y")
    # 检测 tokenizer 类型
    tokenizer_name = str(tokenizer.__class__.__name__).lower()
    tokenizer_path = getattr(tokenizer, 'name_or_path', '').lower()
    
    is_qwen2vl = (
        'qwen' in tokenizer_name or 
        'qwen' in tokenizer_path
    )
    
    if is_qwen2vl and not force_legacy:
        # ============ Qwen2-VL 处理逻辑 ============
        prompt_parts = prompt.split('<image>')
        
        try:
            vision_start_id = tokenizer.convert_tokens_to_ids('<|vision_start|>')
            vision_end_id = tokenizer.convert_tokens_to_ids('<|vision_end|>')
            image_pad_id = tokenizer.convert_tokens_to_ids('<|image_pad|>')
        except (KeyError, AttributeError):
            # 使用硬编码的 token IDs 作为 fallback
            vision_start_id = 151652
            vision_end_id = 151653
            image_pad_id = 151655
            
        input_ids = []
        
        for i, part in enumerate(prompt_parts):
            if part:
                part_ids = tokenizer(part, add_special_tokens=False).input_ids
                input_ids.extend(part_ids)
            
            if i < len(prompt_parts) - 1:
                # 插入: <|vision_start|> <|image_pad|> <|vision_end|>
                input_ids.extend([vision_start_id, image_pad_id, vision_end_id])
        
        # 确保添加 BOS token
        if tokenizer.bos_token_id is not None and (not input_ids or input_ids[0] != tokenizer.bos_token_id):
            input_ids.insert(0, tokenizer.bos_token_id)
    
    else:
        # ============ LLaVA 处理逻辑（保持不变）============
        prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split('<image>')]

        def insert_separator(X, sep):
            return [ele for sublist in zip(X, [sep]*len(X)) for ele in sublist][:-1]

        input_ids = []
        offset = 0
        if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[0][0] == tokenizer.bos_token_id:
            offset = 1
            input_ids.append(prompt_chunks[0][0])

        for x in insert_separator(prompt_chunks, [image_token_index] * (offset + 1)):
            input_ids.extend(x[offset:])

    if return_tensors is not None:
        if return_tensors == 'pt':
            return torch.tensor(input_ids, dtype=torch.long)
        raise ValueError(f'Unsupported tensor type: {return_tensors}')
    return input_ids
    
def _add_speaker_and_signal(header, source, get_conversation=True):
    """Add speaker and start/end signal on each round."""
    BEGIN_SIGNAL = "### "
    END_SIGNAL = "\n"
    conversation = header
    for sentence in source:
        from_str = sentence["from"]
        if from_str.lower() == "human":
            from_str = conversation_lib.default_conversation.roles[0]
        elif from_str.lower() == "gpt":
            from_str = conversation_lib.default_conversation.roles[1]
        else:
            from_str = 'unknown'
        sentence["value"] = (BEGIN_SIGNAL + from_str + ": " +
                             sentence["value"] + END_SIGNAL)
        if get_conversation:
            conversation += sentence["value"]
    conversation += BEGIN_SIGNAL
    return conversation

def _tokenize_fn(strings: Sequence[str],
                 tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ) for text in strings
    ]
    input_ids = labels = [
        tokenized.input_ids[0] for tokenized in tokenized_list
    ]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )

def _mask_targets(target, tokenized_lens, speakers):
    # cur_idx = 0
    cur_idx = tokenized_lens[0]
    tokenized_lens = tokenized_lens[1:]
    target[:cur_idx] = IGNORE_INDEX
    for tokenized_len, speaker in zip(tokenized_lens, speakers):
        if speaker == "human":
            target[cur_idx+2:cur_idx + tokenized_len] = IGNORE_INDEX
        cur_idx += tokenized_len

def preprocess_llama_2(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.LLAMA_2

    # Mask targets
    sep = "[/INST] "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())
        
        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                # 图像+文本模式：动态计算特殊 token
                rou_tokens_img = tokenizer_image_token(rou, tokenizer, return_tensors=None)
                rou_text_tokens_with = tokenizer(rou, add_special_tokens=True).input_ids
                rou_text_tokens_without = tokenizer(rou, add_special_tokens=False).input_ids
                rou_num_special = len(rou_text_tokens_with) - len(rou_text_tokens_without)
                round_len = len(rou_tokens_img) - rou_num_special
                
                inst_tokens_with_img = tokenizer_image_token(parts[0], tokenizer, return_tensors=None)
                inst_text_tokens_with = tokenizer(parts[0], add_special_tokens=True).input_ids
                inst_text_tokens_without = tokenizer(parts[0], add_special_tokens=False).input_ids
                inst_num_special = len(inst_text_tokens_with) - len(inst_text_tokens_without)
                instruction_len = len(inst_tokens_with_img) - inst_num_special
                
                num_special_tokens = inst_num_special
            else:
                # 动态计算特殊 token 数量，适配不同 tokenizer (LLaMA/Qwen)
                rou_tokens_with = tokenizer(rou, add_special_tokens=True).input_ids
                rou_tokens_without = tokenizer(rou, add_special_tokens=False).input_ids
                rou_num_special = len(rou_tokens_with) - len(rou_tokens_without)
                round_len = len(rou_tokens_with) - rou_num_special
                
                inst_tokens_with = tokenizer(parts[0], add_special_tokens=True).input_ids
                inst_tokens_without = tokenizer(parts[0], add_special_tokens=False).input_ids
                inst_num_special = len(inst_tokens_with) - len(inst_tokens_without)
                instruction_len = len(inst_tokens_with) - inst_num_special
                
                num_special_tokens = inst_num_special
            
            # 动态修复：在第一轮时自动调整 cur_len（适配不同 tokenizer）
            if i == 0:
                expected_cur_len = total_len - round_len
                if cur_len != expected_cur_len:
                    target[:expected_cur_len] = IGNORE_INDEX
                    cur_len = expected_cur_len

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                # 详细调试信息
                debug_info = f"has_image={has_image}, num_special={num_special_tokens if 'num_special_tokens' in locals() else 'N/A'}, "
                debug_info += f"round_len={round_len if 'round_len' in locals() else 'N/A'}, "
                debug_info += f"instruction_len={instruction_len if 'instruction_len' in locals() else 'N/A'}"
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}. (ignored)\n"
                    f"  DEBUG: {debug_info}\n"
                    f"  Conversation snippet: {conversation[:100] if len(conversation) > 0 else 'empty'}..."
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_v1(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False,
    training_mode: bool =True,
    only_one_system_prompt = False,
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())
        if only_one_system_prompt:
            if conv.system != '':  # for multi round conversations
                conv.system = ''

    # Tokenize conversations

    if has_image:
        if training_mode:
            input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
        else:
            input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
            return dict(
                input_ids=input_ids,
            )
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids
    
    input_ids = input_ids[:, :tokenizer.model_max_length]
    
    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO

    # Mask targets
    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        
        # ============================================
        # 🔧 步骤 1: 预先计算所有轮次的长度
        # ============================================
        round_lengths = []
        instruction_lengths = []
        
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                # 图像+文本模式：动态计算特殊 token
                rou_tokens_img = tokenizer_image_token(rou, tokenizer, return_tensors=None)
                rou_text_tokens_with = tokenizer(rou, add_special_tokens=True).input_ids
                rou_text_tokens_without = tokenizer(rou, add_special_tokens=False).input_ids
                rou_num_special = len(rou_text_tokens_with) - len(rou_text_tokens_without)
                round_len = len(rou_tokens_img) - rou_num_special
                
                inst_tokens_with_img = tokenizer_image_token(parts[0], tokenizer, return_tensors=None)
                inst_text_tokens_with = tokenizer(parts[0], add_special_tokens=True).input_ids
                inst_text_tokens_without = tokenizer(parts[0], add_special_tokens=False).input_ids
                inst_num_special = len(inst_text_tokens_with) - len(inst_text_tokens_without)
                instruction_len = len(inst_tokens_with_img) - inst_num_special
                
                num_special_tokens = inst_num_special
            else:
                # 动态计算特殊 token 数量，适配不同 tokenizer (LLaMA/Qwen)
                rou_tokens_with = tokenizer(rou, add_special_tokens=True).input_ids
                rou_tokens_without = tokenizer(rou, add_special_tokens=False).input_ids
                rou_num_special = len(rou_tokens_with) - len(rou_tokens_without)
                round_len = len(rou_tokens_with) - rou_num_special
                
                inst_tokens_with = tokenizer(parts[0], add_special_tokens=True).input_ids
                inst_tokens_without = tokenizer(parts[0], add_special_tokens=False).input_ids
                inst_num_special = len(inst_tokens_with) - len(inst_tokens_without)
                instruction_len = len(inst_tokens_with) - inst_num_special
                
                num_special_tokens = inst_num_special
            
            round_lengths.append(round_len)
            instruction_lengths.append(instruction_len)
        
        # ============================================
        # 🔧 步骤 2: 修正第一轮的 cur_len
        # ============================================
        if len(round_lengths) > 0:
            expected_cur_len = total_len - sum(round_lengths)
            if cur_len != expected_cur_len:
                target[:expected_cur_len] = IGNORE_INDEX
                cur_len = expected_cur_len
        
        # ============================================
        # 🔧 步骤 3: 逐轮处理，在最后一轮进行精确修正
        # ============================================
        num_valid_rounds = len(round_lengths)
        for i in range(num_valid_rounds):
            round_len = round_lengths[i]
            instruction_len = instruction_lengths[i]
            
            # 关键修复：在最后一轮使用精确的剩余长度
            if i == num_valid_rounds - 1:
                remaining_len = total_len - cur_len
                if remaining_len != round_len:
                    round_len = remaining_len
                    if instruction_len > round_len:
                        instruction_len = max(0, round_len - 1)
            
            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX
            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                if len(rounds) > 1:
                    print(
                        f"ERROR: tokenization mismatch after fix: {cur_len} vs. {total_len}. "
                        f"This should not happen! Sample IGNORED.\n"
                        f"  num_rounds={num_valid_rounds}, "
                        f"  Conversation snippet: {conversation[:100]}..."
                    )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_mpt(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations
    input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    targets = input_ids.clone()
    assert conv.sep_style == conversation_lib.SeparatorStyle.MPT

    # Mask targets
    sep = conv.sep + conv.roles[1]
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep)
        re_rounds = [conv.sep.join(rounds[:3])] # system + user + gpt
        for conv_idx in range(3, len(rounds), 2):
            re_rounds.append(conv.sep.join(rounds[conv_idx:conv_idx+2]))    # user + gpt
        cur_len = 0
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(re_rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep
            round_len = len(tokenizer_image_token(rou, tokenizer)) + len(tokenizer_image_token(conv.sep, tokenizer))
            instruction_len = len(tokenizer_image_token(parts[0], tokenizer))
            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_plain(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        assert len(source) == 2
        assert DEFAULT_IMAGE_TOKEN in source[0]['value']
        source[0]['value'] = DEFAULT_IMAGE_TOKEN
        conversation = source[0]['value'] + source[1]['value'] + conversation_lib.default_conversation.sep
        conversations.append(conversation)
    # tokenize conversations
    input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        tokenized_len = len(tokenizer_image_token(source[0]['value'], tokenizer))
        target[:tokenized_len] = IGNORE_INDEX

    return dict(input_ids=input_ids, labels=targets)

def preprocess(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False,
    training_mode: bool =True,
    only_one_system_prompt = False,
) -> Dict:
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.PLAIN:
        return preprocess_plain(sources, tokenizer)
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.LLAMA_2:
        return preprocess_llama_2(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version.startswith("v1"):
        return preprocess_v1(sources, tokenizer, has_image=has_image, training_mode=training_mode, only_one_system_prompt=only_one_system_prompt)
    if conversation_lib.default_conversation.version == "mpt":
        return preprocess_mpt(sources, tokenizer)
    
    conversations = []
    for source in sources:
        header = f"{conversation_lib.default_conversation.system}\n\n"
        conversation = _add_speaker_and_signal(header, source)
        conversations.append(conversation)
    # tokenize conversations
    def get_tokenize_len(prompts):
        return [min(len(tokenizer_image_token(prompt, tokenizer)), tokenizer.model_max_length) for prompt in prompts]

    if has_image:
        input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt')[:tokenizer.model_max_length] for prompt in conversations]
    else:
        conversations_tokenized = _tokenize_fn(conversations, tokenizer)
        input_ids = conversations_tokenized["input_ids"]

    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        if has_image:
            tokenized_lens = get_tokenize_len([header] + [s["value"] for s in source])
        else:
            tokenized_lens = _tokenize_fn([header] + [s["value"] for s in source], tokenizer)["input_ids_lens"]
        speakers = [sentence["from"] for sentence in source]
        _mask_targets(target, tokenized_lens, speakers)

    return dict(input_ids=input_ids, labels=targets)