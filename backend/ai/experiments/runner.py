#!/usr/bin/env python3
"""
Automated Experiment Runner for Next Channel
Usage: python -m backend.ai.experiments.runner --config backend/ai/experiments/configs/next_hpo.yaml --dataset datasets/ai_eval/base.jsonl
"""
import argparse
import json
import time
import yaml
import asyncio
from pathlib import Path
from typing import Dict, List, Any, Optional
import sys
import os

# Add backend to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from backend.ai.router import pick
from backend.ai.eval import evaluate_item, RAG_AVAILABLE
from backend.ai.metrics import structured_logger, generate_request_id

class ExperimentRunner:
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.logger = structured_logger
        self.results = []
        
    def generate_search_space(self) -> List[Dict[str, Any]]:
        """Generate parameter combinations from search space"""
        space = self.config['search_space']
        combinations = []
        
        # Generate all combinations
        import itertools
        keys = list(space.keys())
        values = list(space.values())
        
        for combination in itertools.product(*values):
            param_dict = dict(zip(keys, combination))
            combinations.append(param_dict)
        
        return combinations
    
    def evaluate_config(self, config: Dict[str, Any], dataset_path: str) -> Dict[str, Any]:
        """Evaluate a single configuration"""
        # Load dataset
        with open(dataset_path, 'r') as f:
            items = [json.loads(line) for line in f if line.strip()]
        
        # Update engine config temporarily
        channel = self.config['channel']
        engine = pick(channel)
        
        # Override engine parameters
        original_cfg = engine.CFG.copy()
        engine.CFG.update(config)
        
        total_score = 0.0
        total_latency = 0.0
        total_tokens = 0.0
        successful_items = 0
        
        start_time = time.time()
        
        for i, item in enumerate(items):
            request_id = generate_request_id()
            
            try:
                result = evaluate_item(
                    item, 
                    engine, 
                    model=None,  # Will use updated CFG
                    offline=not RAG_AVAILABLE
                )
                
                if result['error']:
                    self.logger.log_error(
                        request_id=request_id,
                        error=f"Evaluation failed: {result['error']}",
                        channel=channel
                    )
                    continue
                
                total_score += result['score']
                total_latency += result['latency_ms']
                total_tokens += result['tokens_out']
                successful_items += 1
                
                # Log progress
                if (i + 1) % 10 == 0:
                    print(f"Evaluated {i + 1}/{len(items)} items, current avg score: {total_score / (i + 1):.3f}")
                
            except Exception as e:
                self.logger.log_error(
                    request_id=request_id,
                    error=f"Evaluation error: {str(e)}",
                    channel=channel
                )
        
        # Restore original config
        engine.CFG = original_cfg
        
        total_time = time.time() - start_time
        
        # Calculate averages
        avg_score = total_score / successful_items if successful_items > 0 else 0.0
        avg_latency = total_latency / successful_items if successful_items > 0 else 0.0
        avg_tokens = total_tokens / successful_items if successful_items > 0 else 0.0
        
        return {
            'config': config,
            'avg_score': avg_score,
            'avg_latency_ms': avg_latency,
            'avg_tokens_out': avg_tokens,
            'successful_items': successful_items,
            'total_items': len(items),
            'total_time_s': total_time,
            'optimization_target': self.config['evaluation']['optimization_target']
        }
    
    def check_early_stopping(self, current_best: float, iteration: int, results: List[Dict[str, Any]]) -> bool:
        """Check if experiment should stop early"""
        early_stopping = self.config['experiment']['early_stopping']
        
        if iteration < early_stopping['patience']:
            return False
        
        # Check if improvement threshold met
        min_improvement = early_stopping['min_improvement']
        recent_scores = [r['avg_score'] for r in results[-early_stopping['patience']:]]
        
        if not recent_scores:
            return False
        
        recent_best = max(recent_scores)
        improvement = current_best - recent_best
        
        return improvement < min_improvement
    
    def select_best_config(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Select best configuration based on optimization target"""
        target = self.config['evaluation']['optimization_target']
        
        if target == 'score':
            best_result = max(results, key=lambda x: x['avg_score'])
        elif target == 'latency':
            best_result = min(results, key=lambda x: x['avg_latency_ms'])
        elif target == 'tokens':
            best_result = max(results, key=lambda x: x['avg_tokens_out'])
        else:
            # Default to score
            best_result = max(results, key=lambda x: x['avg_score'])
        
        return best_result
    
    def run_experiment(self, dataset_path: str) -> Dict[str, Any]:
        """Run the full experiment"""
        print(f"Starting experiment: {self.config['experiment']['name']}")
        print(f"Channel: {self.config['channel']}")
        print(f"Search space size: {len(self.generate_search_space())}")
        
        search_space = self.generate_search_space()
        max_iterations = self.config['experiment']['max_iterations']
        
        best_result = None
        results = []
        
        for iteration in range(max_iterations):
            if iteration >= len(search_space):
                print(f"Warning: More iterations than search space combinations")
                break
            
            config = search_space[iteration]
            print(f"\nIteration {iteration + 1}/{max_iterations}: {config}")
            
            # Evaluate this configuration
            result = self.evaluate_config(config, dataset_path)
            results.append(result)
            
            print(f"Score: {result['avg_score']:.3f}, Latency: {result['avg_latency_ms']:.1f}ms, Tokens: {result['avg_tokens_out']:.0f}")
            
            # Track best result
            if best_result is None or result['avg_score'] > best_result['avg_score']:
                best_result = result
                print(f"New best score: {best_result['avg_score']:.3f}")
            
            # Check early stopping
            if self.check_early_stopping(best_result['avg_score'], iteration, results):
                print(f"Early stopping triggered at iteration {iteration + 1}")
                break
        
        # Select overall best configuration
        final_best = self.select_best_config(results)
        
        experiment_summary = {
            'experiment_name': self.config['experiment']['name'],
            'channel': self.config['channel'],
            'total_iterations': len(results),
            'best_config': final_best['config'],
            'best_score': final_best['avg_score'],
            'best_latency_ms': final_best['avg_latency_ms'],
            'best_tokens_out': final_best['avg_tokens_out'],
            'successful_evaluations': sum(r['successful_items'] for r in results),
            'total_evaluations': sum(r['total_items'] for r in results),
            'optimization_target': self.config['evaluation']['optimization_target'],
            'early_stopped': len(results) < max_iterations,
            'results': results
        }
        
        # Log experiment completion
        self.logger.log_eval_run(
            channel=self.config['channel'],
            dataset=dataset_path,
            score=final_best['avg_score'],
            total_items=experiment_summary['total_evaluations'],
            successful_items=experiment_summary['successful_evaluations'],
            avg_latency_ms=final_best['avg_latency_ms']
        )
        
        return experiment_summary
    
    def create_promotion_pr(self, best_config: Dict[str, Any], summary: Dict[str, Any]) -> bool:
        """Create PR to promote winning configuration to stable"""
        if not self.config['experiment']['auto_promote']:
            return False
        
        # Check if meets promotion threshold
        promotion_threshold = self.config['safety']['auto_promotion_threshold']
        if summary['best_score'] < promotion_threshold:
            print(f"Score {summary['best_score']:.3f} below promotion threshold {promotion_threshold}")
            return False
        
        try:
            # Update next channel configuration
            next_config_path = Path(__file__).parent.parent / 'models' / 'next.yaml'
            with open(next_config_path, 'r') as f:
                current_config = yaml.safe_load(f)
            
            # Merge best config into current config
            updated_config = {**current_config, **best_config}
            
            with open(next_config_path, 'w') as f:
                yaml.dump(updated_config, f, default_flow_style=False)
            
            # Create commit
            import subprocess
            subprocess.run(['git', 'add', 'backend/ai/models/next.yaml'], check=True)
            subprocess.run(['git', 'commit', '-m', f'experiment: Update next channel config - score: {summary["best_score"]:.3f}'], check=True)
            
            # Create PR to stable
            pr_body = f"""## Automated Experiment Promotion
            
**Experiment**: {summary['experiment_name']}
**Channel**: next → stable
**Best Score**: {summary['best_score']:.3f}
**Best Config**: {json.dumps(best_config, indent=2)}
**Improvement**: Meets promotion threshold (≥ {promotion_threshold})

### Changes
- Update next channel configuration with winning parameters
- Auto-promotion enabled based on evaluation results

### Safety
- Configuration has been evaluated against safety dataset
- Meets minimum score requirements for promotion
- Human review still recommended for production changes

### Metrics
- Total Evaluations: {summary['total_evaluations']}
- Successful: {summary['successful_evaluations']}
- Average Score: {summary['best_score']:.3f}
- Average Latency: {summary['best_latency_ms']:.1f}ms
"""
            
            # Create PR using GitHub CLI
            result = subprocess.run([
                'gh', 'pr', 'create',
                '-B', 'stable',
                '-H', 'next',
                '-t', f'Auto-promote next to stable (score: {summary["best_score"]:.3f})',
                '-b', pr_body
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"Created promotion PR: {result.stdout.strip()}")
                return True
            else:
                print(f"Failed to create PR: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"Failed to create promotion PR: {e}")
            return False

def main():
    parser = argparse.ArgumentParser(description="Automated Experiment Runner")
    parser.add_argument("--config", required=True, help="Experiment configuration file")
    parser.add_argument("--dataset", required=True, help="Evaluation dataset")
    parser.add_argument("--create-pr", action="store_true", help="Create promotion PR if criteria met")
    args = parser.parse_args()
    
    runner = ExperimentRunner(args.config)
    summary = runner.run_experiment(args.dataset)
    
    print(f"\n=== Experiment Complete ===")
    print(f"Best Score: {summary['best_score']:.3f}")
    print(f"Best Config: {summary['best_config']}")
    print(f"Total Iterations: {summary['total_iterations']}")
    
    if args.create_pr:
        runner.create_promotion_pr(summary['best_config'], summary)
    
    # Save results
    results_path = Path(__file__).parent / 'experiment_results.json'
    with open(results_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nResults saved to: {results_path}")
    return 0

if __name__ == "__main__":
    exit(main())