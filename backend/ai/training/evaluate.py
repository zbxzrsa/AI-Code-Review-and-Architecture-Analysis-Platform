import argparse
import json
from typing import Dict, List, Tuple, Set


def load_jsonl(path: str) -> List[Dict]:
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data


def set_from_items(items: List[Dict], key: str) -> Set[str]:
    return set([str(x.get(key)) for x in items if key in x])


def compute_prf(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def eval_defects(gt: List[Dict], pred: List[Dict]) -> Dict[str, float]:
    gt_map = {d['id']: d for d in gt}
    pred_map = {d['id']: d for d in pred}
    tp = fp = fn = 0
    ids = set(gt_map.keys()) | set(pred_map.keys())
    for i in ids:
        gt_defs = set_from_items(gt_map.get(i, {}).get('defects', []), 'defect_type')
        pr_defs = set_from_items(pred_map.get(i, {}).get('defects', []), 'defect_type')
        tp += len(gt_defs & pr_defs)
        fp += len(pr_defs - gt_defs)
        fn += len(gt_defs - pr_defs)
    precision, recall, f1 = compute_prf(tp, fp, fn)
    # 微平均“准确率”近似：TP / (TP+FP+FN)
    accuracy = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
    fp_rate = fp / (tp + fp) if (tp + fp) > 0 else 0.0
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'false_positive_rate': fp_rate,
    }


def eval_arch_patterns(gt: List[Dict], pred: List[Dict]) -> Dict[str, float]:
    gt_map = {d['id']: d for d in gt}
    pred_map = {d['id']: d for d in pred}
    tp = fp = fn = 0
    ids = set(gt_map.keys()) | set(pred_map.keys())
    for i in ids:
        gt_patterns = set_from_items(gt_map.get(i, {}).get('patterns', []), 'pattern_name')
        pr_patterns = set_from_items(pred_map.get(i, {}).get('patterns', []), 'pattern_name')
        tp += len(gt_patterns & pr_patterns)
        fp += len(pr_patterns - gt_patterns)
        fn += len(gt_patterns - pr_patterns)
    precision, recall, f1 = compute_prf(tp, fp, fn)
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate AI analysis metrics against targets')
    parser.add_argument('--defects-gt', type=str, help='Ground truth defects JSONL')
    parser.add_argument('--defects-pred', type=str, help='Predicted defects JSONL')
    parser.add_argument('--arch-gt', type=str, help='Ground truth architecture patterns JSONL')
    parser.add_argument('--arch-pred', type=str, help='Predicted architecture patterns JSONL')
    parser.add_argument('--out', type=str, default=None, help='Output JSON summary path')
    args = parser.parse_args()

    summary: Dict[str, Dict[str, float]] = {}

    if args.defects_gt and args.defects_pred:
        defects_gt = load_jsonl(args.defects_gt)
        defects_pred = load_jsonl(args.defects_pred)
        summary['defects'] = eval_defects(defects_gt, defects_pred)
        summary['defects']['meets_accuracy_target'] = summary['defects']['accuracy'] >= 0.85
        summary['defects']['meets_fp_target'] = summary['defects']['false_positive_rate'] <= 0.15

    if args.arch_gt and args.arch_pred:
        arch_gt = load_jsonl(args.arch_gt)
        arch_pred = load_jsonl(args.arch_pred)
        summary['architecture'] = eval_arch_patterns(arch_gt, arch_pred)
        summary['architecture']['meets_f1_target'] = summary['architecture']['f1'] >= 0.8

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.out:
        with open(args.out, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    main()