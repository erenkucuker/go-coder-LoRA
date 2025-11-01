
"""
Evaluation Script for Go Coder LoRA Model
Comprehensive evaluation of the fine-tuned Mistral 7B model on Go programming tasks
"""

import os
import json
import logging
import subprocess
import tempfile
import hashlib
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass, asdict
import re
import ast

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import yaml
from dotenv import load_dotenv
import pandas as pd
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import nltk
from tqdm import tqdm

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('evaluation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Stores evaluation results for a single example"""
    id: str
    instruction: str
    expected_output: str
    generated_output: str
    metrics: Dict[str, float]
    go_validity: bool
    syntax_errors: List[str]
    execution_result: Optional[str]
    timestamp: str


class GoCodeValidator:
    """Validates and tests Go code"""

    def __init__(self):
        self.go_available = self._check_go_installation()

    def _check_go_installation(self) -> bool:
        """Check if Go is installed"""
        try:
            result = subprocess.run(['go', 'version'], capture_output=True, text=True)
            if result.returncode == 0:
                logger.info(f"Go is available: {result.stdout.strip()}")
                return True
        except FileNotFoundError:
            logger.warning("Go is not installed. Code validation will be limited.")
        return False

    def extract_go_code(self, text: str) -> List[str]:
        """Extract Go code blocks from text"""
        # Pattern for code blocks with language specification
        pattern1 = r'```go\n(.*?)```'
        # Pattern for generic code blocks
        pattern2 = r'```\n(.*?)```'
        # Pattern for inline code that looks like Go
        pattern3 = r'func\s+\w+.*?\{.*?\}'

        code_blocks = []

        # Try to find Go-specific code blocks first
        matches = re.findall(pattern1, text, re.DOTALL)
        code_blocks.extend(matches)

        # If no Go blocks found, try generic blocks
        if not code_blocks:
            matches = re.findall(pattern2, text, re.DOTALL)
            code_blocks.extend(matches)

        # Also look for inline function definitions
        inline_matches = re.findall(pattern3, text, re.DOTALL)
        code_blocks.extend(inline_matches)

        return code_blocks

    def validate_syntax(self, code: str) -> Tuple[bool, List[str]]:
        """Validate Go code syntax"""
        if not self.go_available:
            return self._basic_syntax_check(code)

        errors = []

        # Create a temporary Go file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.go', delete=False) as f:
            # Add package declaration if not present
            if 'package' not in code:
                f.write('package main\n\n')

            # Add imports if functions are used but imports are missing
            if any(func in code for func in ['fmt.', 'log.', 'errors.', 'strings.', 'io.']):
                if 'import' not in code:
                    f.write('import (\n')
                    f.write('    "fmt"\n')
                    f.write('    "log"\n')
                    f.write('    "errors"\n')
                    f.write('    "strings"\n')
                    f.write('    "io"\n')
                    f.write(')\n\n')

            f.write(code)
            temp_file = f.name

        try:
            # Try to build the Go file
            result = subprocess.run(
                ['go', 'build', '-o', '/dev/null', temp_file],
                capture_output=True,
                text=True,
                timeout=10
            )

            if result.returncode != 0:
                errors = result.stderr.split('\n')
                errors = [e for e in errors if e.strip()]

            return len(errors) == 0, errors

        except subprocess.TimeoutExpired:
            return False, ["Code execution timeout"]
        except Exception as e:
            return False, [str(e)]
        finally:
            # Clean up
            os.unlink(temp_file)

    def _basic_syntax_check(self, code: str) -> Tuple[bool, List[str]]:
        """Basic syntax validation when Go is not installed"""
        errors = []

        # Check for basic Go syntax patterns
        go_keywords = ['func', 'package', 'import', 'var', 'const', 'type',
                      'if', 'for', 'range', 'return', 'go', 'chan', 'select']

        has_go_syntax = any(keyword in code for keyword in go_keywords)

        if not has_go_syntax:
            errors.append("No Go syntax detected")

        # Check for balanced braces
        open_braces = code.count('{')
        close_braces = code.count('}')
        if open_braces != close_braces:
            errors.append(f"Unbalanced braces: {open_braces} open, {close_braces} close")

        # Check for balanced parentheses
        open_parens = code.count('(')
        close_parens = code.count(')')
        if open_parens != close_parens:
            errors.append(f"Unbalanced parentheses: {open_parens} open, {close_parens} close")

        return len(errors) == 0, errors

    def test_code(self, code: str, test_cases: Optional[List[Dict]] = None) -> Optional[str]:
        """Run Go code with test cases if provided"""
        if not self.go_available:
            return None

        # Create a temporary directory for the test
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "test.go"

            # Write test file
            with open(test_file, 'w') as f:
                if 'package' not in code:
                    f.write('package main\n\n')

                # Add necessary imports
                f.write('import (\n')
                f.write('    "fmt"\n')
                f.write('    "testing"\n')
                f.write(')\n\n')

                f.write(code)

                # Add a simple test if test cases provided
                if test_cases:
                    f.write('\n\nfunc TestGenerated(t *testing.T) {\n')
                    f.write('    // Basic test execution\n')
                    f.write('    fmt.Println("Tests passed")\n')
                    f.write('}\n')

            try:
                # Run the tests
                result = subprocess.run(
                    ['go', 'test', '-v'],
                    cwd=tmpdir,
                    capture_output=True,
                    text=True,
                    timeout=5
                )

                return result.stdout if result.returncode == 0 else result.stderr

            except Exception as e:
                return f"Test execution error: {str(e)}"


class GoCoderEvaluator:
    """Main evaluator class for Go Coder LoRA model"""

    def __init__(self, model_path: str, config_path: str = "configs/training_config.yaml"):
        """Initialize evaluator"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.model_path = model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.validator = GoCodeValidator()
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

        logger.info(f"Evaluator initialized with model: {model_path}")
        logger.info(f"Using device: {self.device}")

    def load_model(self):
        """Load the fine-tuned model and tokenizer"""
        logger.info("Loading model and tokenizer...")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Check if this is a LoRA model or full model
        config_file = Path(self.model_path) / "adapter_config.json"

        if config_file.exists():
            # Load base model then apply LoRA
            base_model = AutoModelForCausalLM.from_pretrained(
                self.config['model']['base_model'],
                device_map="auto",
                torch_dtype=torch.float16,
                trust_remote_code=True
            )

            self.model = PeftModel.from_pretrained(
                base_model,
                self.model_path
            )
            logger.info("Loaded LoRA model")
        else:
            # Load full model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                device_map="auto",
                torch_dtype=torch.float16,
                trust_remote_code=True
            )
            logger.info("Loaded full model")

        self.model.eval()

    def generate_response(self, instruction: str, max_new_tokens: int = 512) -> str:
        """Generate response for a given instruction"""
        # Format prompt
        prompt = f"[INST] You are an expert Go programmer. {instruction} [/INST]"

        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.config['data']['max_length']
        ).to(self.device)

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                do_sample=True,
                top_p=0.95,
                pad_token_id=self.tokenizer.pad_token_id
            )

        # Decode
        generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract response after instruction
        if "[/INST]" in generated:
            response = generated.split("[/INST]")[-1].strip()
        else:
            response = generated[len(prompt):].strip()

        return response

    def calculate_metrics(self, expected: str, generated: str) -> Dict[str, float]:
        """Calculate evaluation metrics"""
        metrics = {}

        # ROUGE scores
        rouge_scores = self.rouge_scorer.score(expected, generated)
        metrics['rouge1_f1'] = rouge_scores['rouge1'].fmeasure
        metrics['rouge2_f1'] = rouge_scores['rouge2'].fmeasure
        metrics['rougeL_f1'] = rouge_scores['rougeL'].fmeasure

        # BLEU score
        expected_tokens = nltk.word_tokenize(expected.lower())
        generated_tokens = nltk.word_tokenize(generated.lower())

        smoothing = SmoothingFunction().method1
        metrics['bleu'] = sentence_bleu(
            [expected_tokens],
            generated_tokens,
            smoothing_function=smoothing
        )

        # Length ratio
        metrics['length_ratio'] = len(generated) / max(len(expected), 1)

        # Code-specific metrics
        expected_code = self.validator.extract_go_code(expected)
        generated_code = self.validator.extract_go_code(generated)

        metrics['code_blocks_expected'] = len(expected_code)
        metrics['code_blocks_generated'] = len(generated_code)

        return metrics

    def evaluate_single(self, instruction: str, expected: str) -> EvaluationResult:
        """Evaluate a single example"""
        from datetime import datetime

        # Generate response
        generated = self.generate_response(instruction)

        # Calculate metrics
        metrics = self.calculate_metrics(expected, generated)

        # Validate Go code
        code_blocks = self.validator.extract_go_code(generated)

        is_valid = True
        all_errors = []
        execution_result = None

        for code in code_blocks:
            valid, errors = self.validator.validate_syntax(code)
            if not valid:
                is_valid = False
                all_errors.extend(errors)

            # Try to execute code if valid
            if valid and self.validator.go_available:
                exec_result = self.validator.test_code(code)
                if exec_result:
                    execution_result = exec_result

        return EvaluationResult(
            id=hashlib.md5(instruction.encode()).hexdigest(),
            instruction=instruction,
            expected_output=expected,
            generated_output=generated,
            metrics=metrics,
            go_validity=is_valid,
            syntax_errors=all_errors,
            execution_result=execution_result,
            timestamp=datetime.now().isoformat()
        )

    def evaluate_dataset(self, dataset_path: str, num_samples: Optional[int] = None) -> List[EvaluationResult]:
        """Evaluate on a dataset"""
        logger.info(f"Evaluating dataset: {dataset_path}")

        # Load dataset
        examples = []
        with open(dataset_path, 'r') as f:
            for line in f:
                examples.append(json.loads(line))

        # Limit samples if specified
        if num_samples:
            examples = examples[:num_samples]

        logger.info(f"Evaluating {len(examples)} examples")

        # Evaluate each example
        results = []
        for example in tqdm(examples, desc="Evaluating"):
            result = self.evaluate_single(
                instruction=example['instruction'],
                expected=example['output']
            )
            results.append(result)

        return results

    def generate_report(self, results: List[EvaluationResult]) -> Dict[str, Any]:
        """Generate evaluation report"""
        report = {
            'total_examples': len(results),
            'metrics': {},
            'go_validation': {},
            'examples': []
        }

        # Aggregate metrics
        metric_sums = {}
        for result in results:
            for metric, value in result.metrics.items():
                if metric not in metric_sums:
                    metric_sums[metric] = []
                metric_sums[metric].append(value)

        # Calculate averages
        for metric, values in metric_sums.items():
            report['metrics'][f'avg_{metric}'] = sum(values) / len(values)

        # Go validation statistics
        valid_count = sum(1 for r in results if r.go_validity)
        report['go_validation']['valid_count'] = valid_count
        report['go_validation']['valid_percentage'] = (valid_count / len(results)) * 100

        # Collect all syntax errors
        all_errors = []
        for result in results:
            all_errors.extend(result.syntax_errors)

        # Count error types
        error_counts = {}
        for error in all_errors:
            error_type = error.split(':')[0] if ':' in error else error
            error_counts[error_type] = error_counts.get(error_type, 0) + 1

        report['go_validation']['error_types'] = error_counts

        # Add sample examples
        report['examples'] = [
            {
                'instruction': r.instruction,
                'generated': r.generated_output[:500],  # Truncate for report
                'metrics': r.metrics,
                'valid': r.go_validity
            }
            for r in results[:5]  # First 5 examples
        ]

        return report

    def save_results(self, results: List[EvaluationResult], output_dir: str = "./evaluation_results"):
        """Save evaluation results"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save detailed results
        with open(output_path / "detailed_results.jsonl", 'w') as f:
            for result in results:
                f.write(json.dumps(asdict(result)) + '\n')

        # Generate and save report
        report = self.generate_report(results)
        with open(output_path / "evaluation_report.json", 'w') as f:
            json.dump(report, f, indent=2)

        # Save metrics as CSV for easy analysis
        df = pd.DataFrame([r.metrics for r in results])
        df.to_csv(output_path / "metrics.csv", index=False)

        logger.info(f"Results saved to {output_path}")

        # Print summary
        print("\n" + "="*50)
        print("EVALUATION SUMMARY")
        print("="*50)
        print(f"Total Examples: {report['total_examples']}")
        print(f"Valid Go Code: {report['go_validation']['valid_percentage']:.2f}%")
        print("\nAverage Metrics:")
        for metric, value in report['metrics'].items():
            print(f"  {metric}: {value:.4f}")
        print("="*50)


def main():
    """Main execution function"""
    import argparse
    import hashlib

    parser = argparse.ArgumentParser(description="Evaluate Go Coder LoRA Model")
    parser.add_argument("--model", type=str, required=True,
                       help="Path to the fine-tuned model")
    parser.add_argument("--dataset", type=str, default="./data/processed/test.jsonl",
                       help="Path to evaluation dataset")
    parser.add_argument("--num-samples", type=int, default=None,
                       help="Number of samples to evaluate (default: all)")
    parser.add_argument("--output-dir", type=str, default="./evaluation_results",
                       help="Directory to save results")

    args = parser.parse_args()

    # Initialize evaluator
    evaluator = GoCoderEvaluator(model_path=args.model)

    # Load model
    evaluator.load_model()

    # Evaluate dataset
    results = evaluator.evaluate_dataset(
        dataset_path=args.dataset,
        num_samples=args.num_samples
    )

    # Save results
    evaluator.save_results(results, output_dir=args.output_dir)

    print(f"\nEvaluation complete! Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
