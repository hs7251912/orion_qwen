#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
VQA评估脚本 - 使用BLEU、ROUGE_L和CIDEr评估生成文本的质量
"""

import os
import json
import argparse
from collections import defaultdict
from pathlib import Path
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize
import nltk

# 下载必要的NLTK数据
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')


class RougeL:
    """ROUGE-L评估指标实现"""
    
    @staticmethod
    def lcs(string1, string2):
        """计算最长公共子序列长度"""
        m = len(string1)
        n = len(string2)
        
        # 创建DP表
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if string1[i-1] == string2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        return dp[m][n]
    
    @staticmethod
    def rouge_l_score(candidate, reference):
        """计算ROUGE-L分数"""
        # 分词
        try:
            candidate_tokens = word_tokenize(candidate.lower())
            reference_tokens = word_tokenize(reference.lower())
        except:
            # 如果分词失败，使用简单的空格分割
            candidate_tokens = candidate.lower().split()
            reference_tokens = reference.lower().split()
        
        if len(candidate_tokens) == 0 or len(reference_tokens) == 0:
            return 0.0
        
        lcs_length = RougeL.lcs(candidate_tokens, reference_tokens)
        
        # 计算precision和recall
        precision = lcs_length / len(candidate_tokens) if len(candidate_tokens) > 0 else 0
        recall = lcs_length / len(reference_tokens) if len(reference_tokens) > 0 else 0
        
        # 计算F1分数
        if precision + recall == 0:
            return 0.0
        
        f1_score = 2 * precision * recall / (precision + recall)
        return f1_score


class CIDErScorer:
    """CIDEr评估指标实现（简化版）"""
    
    def __init__(self, n=4, sigma=6.0):
        self.n = n
        self.sigma = sigma
        
    def compute_doc_freq(self, refs_words):
        """计算文档频率"""
        doc_freq = defaultdict(int)
        for ref_words in refs_words:
            unique_words = set()
            for words in ref_words:
                unique_words.update(words)
            for word in unique_words:
                doc_freq[word] += 1
        return doc_freq
    
    def ngrams(self, words, n):
        """生成n-grams"""
        return [tuple(words[i:i+n]) for i in range(len(words) - n + 1)]
    
    def compute_cider(self, candidate, references):
        """计算CIDEr分数"""
        # 分词
        try:
            candidate_tokens = word_tokenize(candidate.lower())
            references_tokens = [word_tokenize(ref.lower()) for ref in references]
        except:
            candidate_tokens = candidate.lower().split()
            references_tokens = [ref.lower().split() for ref in references]
        
        if len(candidate_tokens) == 0:
            return 0.0
        
        scores = []
        for n in range(1, self.n + 1):
            # 生成n-grams
            candidate_ngrams = self.ngrams(candidate_tokens, n)
            
            if len(candidate_ngrams) == 0:
                continue
            
            # 计算TF-IDF向量
            candidate_vec = defaultdict(float)
            for ngram in candidate_ngrams:
                candidate_vec[ngram] += 1
            
            # 对每个reference计算分数
            ref_scores = []
            for ref_tokens in references_tokens:
                ref_ngrams = self.ngrams(ref_tokens, n)
                if len(ref_ngrams) == 0:
                    ref_scores.append(0.0)
                    continue
                
                ref_vec = defaultdict(float)
                for ngram in ref_ngrams:
                    ref_vec[ngram] += 1
                
                # 计算余弦相似度
                dot_product = sum(candidate_vec[ngram] * ref_vec[ngram] for ngram in candidate_vec)
                candidate_norm = np.sqrt(sum(v**2 for v in candidate_vec.values()))
                ref_norm = np.sqrt(sum(v**2 for v in ref_vec.values()))
                
                if candidate_norm > 0 and ref_norm > 0:
                    score = dot_product / (candidate_norm * ref_norm)
                else:
                    score = 0.0
                
                ref_scores.append(score)
            
            # 取所有reference的平均分
            if ref_scores:
                scores.append(np.mean(ref_scores))
        
        # 返回所有n-gram的平均分
        return np.mean(scores) if scores else 0.0


class VQAEvaluator:
    """VQA评估器"""
    
    def __init__(self, gt_dir, pred_dir):
        self.gt_dir = Path(gt_dir)
        self.pred_dir = Path(pred_dir)
        self.rouge_l = RougeL()
        self.cider = CIDErScorer()
        self.smoothing = SmoothingFunction().method1
        
    def load_json(self, file_path):
        """加载JSON文件"""
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def extract_answers(self, data):
        """从JSON数据中提取所有gpt的回答"""
        answers = []
        for qa_pair in data:
            # 处理两种格式：
            # 格式1: [{"from": "human", "value": "..."}, {"from": "gpt", "value": "..."}]
            # 格式2: {"human": "...", "gpt": "..."}
            if isinstance(qa_pair, list):
                # 标注文件格式（嵌套列表）
                if len(qa_pair) >= 2:
                    gpt_response = qa_pair[1]
                    if gpt_response.get('from') == 'gpt':
                        answers.append(gpt_response.get('value', ''))
            elif isinstance(qa_pair, dict):
                # 预测文件格式（扁平字典）
                if 'gpt' in qa_pair:
                    answers.append(qa_pair.get('gpt', ''))
        return answers
    
    def compute_bleu(self, candidate, reference):
        """计算BLEU分数"""
        try:
            candidate_tokens = word_tokenize(candidate.lower())
            reference_tokens = word_tokenize(reference.lower())
        except:
            candidate_tokens = candidate.lower().split()
            reference_tokens = reference.lower().split()
        
        if len(candidate_tokens) == 0 or len(reference_tokens) == 0:
            return 0.0
        
        # 计算BLEU-4分数
        bleu_score = sentence_bleu(
            [reference_tokens], 
            candidate_tokens,
            smoothing_function=self.smoothing
        )
        return bleu_score
    
    def evaluate_file(self, gt_file, pred_file):
        """评估单个文件"""
        gt_data = self.load_json(gt_file)
        pred_data = self.load_json(pred_file)
        
        gt_answers = self.extract_answers(gt_data)
        pred_answers = self.extract_answers(pred_data)
        
        # 确保答案数量一致
        min_len = min(len(gt_answers), len(pred_answers))
        gt_answers = gt_answers[:min_len]
        pred_answers = pred_answers[:min_len]
        
        if min_len == 0:
            return None
        
        # 计算各种指标
        bleu_scores = []
        rouge_scores = []
        cider_scores = []
        
        for gt_ans, pred_ans in zip(gt_answers, pred_answers):
            # BLEU
            bleu = self.compute_bleu(pred_ans, gt_ans)
            bleu_scores.append(bleu)
            
            # ROUGE-L
            rouge_l = self.rouge_l.rouge_l_score(pred_ans, gt_ans)
            rouge_scores.append(rouge_l)
            
            # CIDEr
            cider = self.cider.compute_cider(pred_ans, [gt_ans])
            cider_scores.append(cider)
        
        return {
            'bleu': np.mean(bleu_scores),
            'rouge_l': np.mean(rouge_scores),
            'cider': np.mean(cider_scores),
            'num_qa': min_len
        }
    
    def evaluate_scene(self, scene_name):
        """评估单个场景"""
        gt_scene_dir = self.gt_dir / scene_name
        pred_scene_dir = self.pred_dir / scene_name
        
        if not gt_scene_dir.exists():
            print(f"警告: Ground truth场景不存在: {scene_name}")
            return None
        
        if not pred_scene_dir.exists():
            print(f"警告: 预测场景不存在: {scene_name}")
            return None
        
        # 获取所有JSON文件
        gt_files = sorted(gt_scene_dir.glob('*.json'))
        
        scene_metrics = {
            'bleu_scores': [],
            'rouge_scores': [],
            'cider_scores': [],
            'num_files': 0,
            'num_qa_total': 0
        }
        
        for gt_file in gt_files:
            pred_file = pred_scene_dir / gt_file.name
            
            if not pred_file.exists():
                print(f"警告: 预测文件不存在: {pred_file}")
                continue
            
            try:
                metrics = self.evaluate_file(gt_file, pred_file)
                if metrics:
                    scene_metrics['bleu_scores'].append(metrics['bleu'])
                    scene_metrics['rouge_scores'].append(metrics['rouge_l'])
                    scene_metrics['cider_scores'].append(metrics['cider'])
                    scene_metrics['num_files'] += 1
                    scene_metrics['num_qa_total'] += metrics['num_qa']
            except Exception as e:
                print(f"错误: 处理文件 {gt_file.name} 时出错: {str(e)}")
                continue
        
        if scene_metrics['num_files'] == 0:
            return None
        
        return {
            'bleu': np.mean(scene_metrics['bleu_scores']),
            'rouge_l': np.mean(scene_metrics['rouge_scores']),
            'cider': np.mean(scene_metrics['cider_scores']),
            'num_files': scene_metrics['num_files'],
            'num_qa': scene_metrics['num_qa_total']
        }
    
    def evaluate_all(self):
        """评估所有场景"""
        # 获取所有场景
        gt_scenes = [d.name for d in self.gt_dir.iterdir() if d.is_dir()]
        pred_scenes = [d.name for d in self.pred_dir.iterdir() if d.is_dir()]
        
        # 只评估两者都存在的场景
        common_scenes = sorted(set(gt_scenes) & set(pred_scenes))
        
        print(f"找到 {len(common_scenes)} 个共同场景")
        print("=" * 80)
        
        all_results = {}
        overall_bleu = []
        overall_rouge = []
        overall_cider = []
        
        for scene_name in common_scenes:
            print(f"\n评估场景: {scene_name}")
            metrics = self.evaluate_scene(scene_name)
            
            if metrics:
                all_results[scene_name] = metrics
                overall_bleu.append(metrics['bleu'])
                overall_rouge.append(metrics['rouge_l'])
                overall_cider.append(metrics['cider'])
                
                print(f"  BLEU:    {metrics['bleu']:.4f}")
                print(f"  ROUGE-L: {metrics['rouge_l']:.4f}")
                print(f"  CIDEr:   {metrics['cider']:.4f}")
                print(f"  文件数:  {metrics['num_files']}")
                print(f"  问答数:  {metrics['num_qa']}")
            else:
                print(f"  跳过 (无有效数据)")
        
        # 计算总体指标
        print("\n" + "=" * 80)
        print("总体评估结果:")
        print("=" * 80)
        if overall_bleu:
            print(f"平均 BLEU:    {np.mean(overall_bleu):.4f} (std: {np.std(overall_bleu):.4f})")
            print(f"平均 ROUGE-L: {np.mean(overall_rouge):.4f} (std: {np.std(overall_rouge):.4f})")
            print(f"平均 CIDEr:   {np.mean(overall_cider):.4f} (std: {np.std(overall_cider):.4f})")
            print(f"评估场景数:   {len(all_results)}")
        
        return all_results


def main():
    parser = argparse.ArgumentParser(description='VQA评估脚本')
    parser.add_argument(
        '--gt_dir',
        type=str,
        default='/root/autodl-tmp/Orion-main/data/chat-B2D/val',
        help='Ground truth数据目录'
    )
    parser.add_argument(
        '--pred_dir',
        type=str,
        default='/root/autodl-tmp/Orion-main/data/chat-B2D/val_answer',
        help='预测结果数据目录'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='evaluation_results.json',
        help='输出结果文件路径'
    )
    
    args = parser.parse_args()
    
    # 创建评估器
    evaluator = VQAEvaluator(args.gt_dir, args.pred_dir)
    
    # 运行评估
    results = evaluator.evaluate_all()
    
    # 保存结果
    output_path = Path(args.output)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n结果已保存到: {output_path}")


if __name__ == '__main__':
    main()

