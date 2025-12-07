import os.path as osp
import pickle
import shutil
import tempfile
import time
import json
import os

import torch
import torch.distributed as dist
from mmcv.image import tensor2imgs
from mmcv.utils import get_dist_info

from mmcv.core import encode_mask_results
from mmcv.fileio.io import dump, load
from mmcv.utils import mkdir_or_exist, ProgressBar

import numpy as np
import pycocotools.mask as mask_util

def save_single_vqa_result(result, data_info, save_root='/root/autodl-tmp/Orion-main/data/chat-B2D/val_answer', verbose=True):
    """
    Save VQA prediction for a single sample immediately after inference.
    
    Args:
        result: Single result dict containing text_out
        data_info: Data info dict containing folder and frame_idx
        save_root: Root directory to save the results
        verbose: If True, print save confirmation for each file
    
    Returns:
        bool: True if saved successfully, False otherwise
    """
    try:
        # Check if text_out exists in result
        if 'text_out' not in result or not result['text_out']:
            return False
        
        # Get scene name and frame ID
        scene_name_raw = data_info.get('folder', 'unknown_scene')
        # Remove 'v1/' or any parent directory prefix, keep only the last part
        scene_name = os.path.basename(scene_name_raw)
        frame_id = data_info.get('frame_idx', 0)
        
        # Create directory for this scene
        scene_dir = osp.join(save_root, scene_name)
        os.makedirs(scene_dir, exist_ok=True)
        
        # Prepare QA pairs
        qa_pairs = []
        for qa_dict in result['text_out']:
            qa_pair = {
                'human': qa_dict['Q'],
                'gpt': qa_dict['A'][0] if isinstance(qa_dict['A'], list) else qa_dict['A']
            }
            qa_pairs.append(qa_pair)
        
        # Save to JSON file with zero-padded frame_id (5 digits)
        json_path = osp.join(scene_dir, f'{frame_id:05d}.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(qa_pairs, f, indent=4, ensure_ascii=False)
        
        if verbose:
            print(f'💾 Saved VQA: {scene_name}/{frame_id:05d}.json ({len(qa_pairs)} QA pairs)')
        
        return True
        
    except Exception as e:
        print(f'❌ Error saving VQA result: {str(e)}')
        return False

def custom_encode_mask_results(mask_results):
    """Encode bitmap mask to RLE code. Semantic Masks only
    Args:
        mask_results (list | tuple[list]): bitmap mask results.
            In mask scoring rcnn, mask_results is a tuple of (segm_results,
            segm_cls_score).
    Returns:
        list | tuple: RLE encoded mask.
    """
    cls_segms = mask_results
    num_classes = len(cls_segms)
    encoded_mask_results = []
    for i in range(len(cls_segms)):
        encoded_mask_results.append(
            mask_util.encode(
                np.array(
                    cls_segms[i][:, :, np.newaxis], order='F',
                        dtype='uint8'))[0])  # encoded with RLE
    return [encoded_mask_results]

def custom_multi_gpu_test(model, data_loader, tmpdir=None, gpu_collect=False):
    """Test model with multiple gpus.
    This method tests model with multiple gpus and collects the results
    under two different modes: gpu and cpu modes. By setting 'gpu_collect=True'
    it encodes results to gpu tensors and use gpu communication for results
    collection. On cpu mode it saves the results on different gpus to 'tmpdir'
    and collects them by the rank 0 worker.
    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.
        tmpdir (str): Path of directory to save the temporary results from
            different gpus under cpu mode.
        gpu_collect (bool): Option to use either gpu or cpu to collect results.
    Returns:
        list: The prediction results.
    """
    model.eval()
    bbox_results = []
    mask_results = []
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = ProgressBar(len(dataset))
    time.sleep(2)  # This line can prevent deadlock problem in some cases.
    have_mask = False
    
    # Counter for saved VQA results (only rank 0 saves)
    vqa_saved_count = 0
    sample_idx = rank  # Each rank starts from its own offset
    
    if rank == 0:
        print(f'\n🚀 Starting distributed inference with real-time VQA saving (rank 0)...')
        print(f'📁 Save directory: /root/autodl-tmp/Orion-main/data/chat-B2D/val_answer')
        print('='*80)
    
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(data,return_loss=False)
            
            # Debug: Check result structure for the first few iterations (only rank 0)
            if rank == 0 and i < 3:
                print(f'\n[DEBUG Rank {rank} Iter {i}] Result type: {type(result)}')
                if isinstance(result, dict):
                    print(f'[DEBUG Rank {rank} Iter {i}] Result keys: {result.keys()}')
                    if 'bbox_results' in result.keys():
                        print(f'[DEBUG Rank {rank} Iter {i}] bbox_results length: {len(result["bbox_results"])}')
                        if len(result['bbox_results']) > 0:
                            print(f'[DEBUG Rank {rank} Iter {i}] First bbox_result keys: {result["bbox_results"][0].keys()}')
                            print(f'[DEBUG Rank {rank} Iter {i}] Has text_out: {"text_out" in result["bbox_results"][0]}')
                elif isinstance(result, list):
                    print(f'[DEBUG Rank {rank} Iter {i}] Result length: {len(result)}')
                    if len(result) > 0:
                        print(f'[DEBUG Rank {rank} Iter {i}] First element type: {type(result[0])}')
                        if isinstance(result[0], dict):
                            print(f'[DEBUG Rank {rank} Iter {i}] First element keys: {result[0].keys()}')
                            print(f'[DEBUG Rank {rank} Iter {i}] Has text_out: {"text_out" in result[0]}')
            
            # encode mask results and save VQA
            if isinstance(result, dict):
                if 'bbox_results' in result.keys():
                    bbox_result = result['bbox_results']
                    batch_size = len(result['bbox_results'])
                    
                    # Save VQA results immediately (only rank 0 to avoid conflicts)
                    if rank == 0:
                        for batch_idx, single_result in enumerate(bbox_result):
                            current_sample_idx = sample_idx + batch_idx * world_size
                            if current_sample_idx < len(dataset.data_infos):
                                data_info = dataset.data_infos[current_sample_idx]
                                # Show verbose output every 10 samples, or always show if less than 50 total
                                verbose = (len(dataset.data_infos) < 50) or (current_sample_idx % 10 == 0)
                                if save_single_vqa_result(single_result, data_info, verbose=verbose):
                                    vqa_saved_count += 1
                                elif verbose:
                                    print(f'⊘ Sample {current_sample_idx}: No VQA output')
                    
                    sample_idx += batch_size * world_size
                    bbox_results.extend(bbox_result)
                if 'mask_results' in result.keys() and result['mask_results'] is not None:
                    mask_result = custom_encode_mask_results(result['mask_results'])
                    mask_results.extend(mask_result)
                    have_mask = True
            else:
                # Result is a list - each element is a dict with bbox results
                batch_size = len(result)
                
                # Save VQA results immediately (only rank 0 to avoid conflicts)
                if rank == 0:
                    for batch_idx, single_result in enumerate(result):
                        current_sample_idx = sample_idx + batch_idx * world_size
                        if current_sample_idx < len(dataset.data_infos):
                            data_info = dataset.data_infos[current_sample_idx]
                            # Show verbose output every 10 samples, or always show if less than 50 total
                            verbose = (len(dataset.data_infos) < 50) or (current_sample_idx % 10 == 0)
                            if save_single_vqa_result(single_result, data_info, verbose=verbose):
                                vqa_saved_count += 1
                            elif verbose:
                                print(f'⊘ Sample {current_sample_idx}: No VQA output')
                
                bbox_results.extend(result)
                sample_idx += batch_size * world_size
        if rank == 0:
            for _ in range(batch_size * world_size):
                prog_bar.update()
    
    if rank == 0:
        print(f'\n✓ Saved {vqa_saved_count} VQA prediction files during inference')
    
    # collect results from all ranks
    if gpu_collect:
        bbox_results = collect_results_gpu(bbox_results, len(dataset))
        if have_mask:
            mask_results = collect_results_gpu(mask_results, len(dataset))
        else:
            mask_results = None
    else:
        bbox_results = collect_results_cpu(bbox_results, len(dataset), tmpdir)
        tmpdir = tmpdir+'_mask' if tmpdir is not None else None
        if have_mask:
            mask_results = collect_results_cpu(mask_results, len(dataset), tmpdir)
        else:
            mask_results = None

    return {'bbox_results': bbox_results, 'mask_results': mask_results}


def collect_results_cpu(result_part, size, tmpdir=None):
    rank, world_size = get_dist_info()
    # create a tmp dir if it is not specified
    if tmpdir is None:
        MAX_LEN = 512
        # 32 is whitespace
        dir_tensor = torch.full((MAX_LEN, ),
                                32,
                                dtype=torch.uint8,
                                device='cuda')
        if rank == 0:
            mkdir_or_exist('.dist_test')
            tmpdir = tempfile.mkdtemp(dir='.dist_test')
            tmpdir = torch.tensor(
                bytearray(tmpdir.encode()), dtype=torch.uint8, device='cuda')
            dir_tensor[:len(tmpdir)] = tmpdir
        dist.broadcast(dir_tensor, 0)
        tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
    else:
        mkdir_or_exist(tmpdir)
    # dump the part result to the dir
    dump(result_part, osp.join(tmpdir, f'part_{rank}.pkl'))
    dist.barrier()
    # collect all parts
    if rank != 0:
        return None
    else:
        # load results of all parts from tmp dir
        part_list = []
        for i in range(world_size):
            part_file = osp.join(tmpdir, f'part_{i}.pkl')
            part_list.append(load(part_file))
        # sort the results
        ordered_results = []
        '''
        bacause we change the sample of the evaluation stage to make sure that each gpu will handle continuous sample,
        '''
        #for res in zip(*part_list):
        for res in part_list:  
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        # remove tmp dir
        shutil.rmtree(tmpdir)
        return ordered_results


def collect_results_gpu(result_part, size):
    collect_results_cpu(result_part, size)


def single_gpu_test(model, data_loader):
    """Test model with single gpu.

    This method tests model with single gpu and gives the 'show' option.
    By setting ``show=True``, it saves the visualization results under
    ``out_dir``.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.
        show (bool, optional): Whether to save viualization results.
            Default: True.
        out_dir (str, optional): The path to save visualization results.
            Default: None.

    Returns:
        list[dict]: The prediction results.
    """
    model.eval()
    bbox_results = []
    mask_results = []
    dataset = data_loader.dataset
    prog_bar = ProgressBar(len(dataset))
    time.sleep(2)  # This line can prevent deadlock problem in some cases.
    have_mask = False
    
    # Counter for saved VQA results
    vqa_saved_count = 0
    sample_idx = 0
    print(f'\n🚀 Starting inference with real-time VQA saving...')
    print(f'📁 Save directory: /root/autodl-tmp/Orion-main/data/chat-B2D/val_answer')
    print('='*80)

    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(data,return_loss=False)
            
            # Debug: Check result structure for the first few iterations
            if i < 3:
                print(f'\n[DEBUG Iter {i}] Result type: {type(result)}')
                if isinstance(result, dict):
                    print(f'[DEBUG Iter {i}] Result keys: {result.keys()}')
                    if 'bbox_results' in result.keys():
                        print(f'[DEBUG Iter {i}] bbox_results length: {len(result["bbox_results"])}')
                        if len(result['bbox_results']) > 0:
                            print(f'[DEBUG Iter {i}] First bbox_result keys: {result["bbox_results"][0].keys()}')
                            print(f'[DEBUG Iter {i}] Has text_out: {"text_out" in result["bbox_results"][0]}')
                elif isinstance(result, list):
                    print(f'[DEBUG Iter {i}] Result length: {len(result)}')
                    if len(result) > 0:
                        print(f'[DEBUG Iter {i}] First element type: {type(result[0])}')
                        if isinstance(result[0], dict):
                            print(f'[DEBUG Iter {i}] First element keys: {result[0].keys()}')
                            print(f'[DEBUG Iter {i}] Has text_out: {"text_out" in result[0]}')

            # encode mask results and save VQA
            if isinstance(result, dict):
                if 'bbox_results' in result.keys():
                    bbox_result = result['bbox_results']
                    batch_size = len(result['bbox_results'])
                    
                    # Save VQA results immediately for each sample in the batch
                    for batch_idx, single_result in enumerate(bbox_result):
                        current_sample_idx = sample_idx + batch_idx
                        if current_sample_idx < len(dataset.data_infos):
                            data_info = dataset.data_infos[current_sample_idx]
                            # Show verbose output every 10 samples, or always show if less than 50 total
                            verbose = (len(dataset.data_infos) < 50) or (current_sample_idx % 10 == 0)
                            if save_single_vqa_result(single_result, data_info, verbose=verbose):
                                vqa_saved_count += 1
                            elif verbose:
                                print(f'⊘ Sample {current_sample_idx}: No VQA output')
                    
                    sample_idx += batch_size
                    bbox_results.extend(bbox_result)
                if 'mask_results' in result.keys() and result['mask_results'] is not None:
                    mask_result = custom_encode_mask_results(result['mask_results'])
                    mask_results.extend(mask_result)
                    have_mask = True
            else:
                # Result is a list - each element is a dict with bbox results
                batch_size = len(result)
                
                # Save VQA results immediately for each sample in the batch
                for batch_idx, single_result in enumerate(result):
                    current_sample_idx = sample_idx + batch_idx
                    if current_sample_idx < len(dataset.data_infos):
                        data_info = dataset.data_infos[current_sample_idx]
                        # Show verbose output every 10 samples, or always show if less than 50 total
                        verbose = (len(dataset.data_infos) < 50) or (current_sample_idx % 10 == 0)
                        if save_single_vqa_result(single_result, data_info, verbose=verbose):
                            vqa_saved_count += 1
                        elif verbose:
                            print(f'⊘ Sample {current_sample_idx}: No VQA output')
                
                bbox_results.extend(result)
                sample_idx += batch_size

            if isinstance(result[0], tuple):
               assert False, 'this code is for instance segmentation, which our code will not utilize.'
               result = [(bbox_results, encode_mask_results(mask_results))
                         for bbox_results, mask_results in result]

        for _ in range(batch_size):
                prog_bar.update()
    
    print(f'\n✓ Saved {vqa_saved_count} VQA prediction files during inference')

    return {'bbox_results': bbox_results, 'mask_results': mask_results}
