"""
LoRA Fine-tuning Script for Mistral 7B - Go Coding Specialization
Implements parameter-efficient fine-tuning with LoRA for Go programming expertise
"""

import os
import sys
import logging
import json
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field

import torch
import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
    EarlyStoppingCallback,
    set_seed
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
from datasets import Dataset, load_dataset
import yaml
from dotenv import load_dotenv
import wandb
from tqdm import tqdm

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """Arguments pertaining to which model/config/tokenizer we are going to fine-tune"""
    model_name_or_path: str = field(
        default="mistralai/Mistral-7B-Instruct-v0.2",
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    use_auth_token: bool = field(
        default=True,
        metadata={"help": "Use HuggingFace auth token"}
    )


@dataclass
class DataArguments:
    """Arguments pertaining to what data we are going to input our model for training and eval"""
    train_data_path: str = field(
        default="./data/processed/train.jsonl",
        metadata={"help": "Path to training dataset"}
    )
    eval_data_path: str = field(
        default="./data/processed/eval.jsonl",
        metadata={"help": "Path to evaluation dataset"}
    )
    max_length: int = field(
        default=2048,
        metadata={"help": "Maximum sequence length"}
    )


@dataclass
class LoRAArguments:
    """Arguments for LoRA configuration"""
    lora_r: int = field(default=64, metadata={"help": "LoRA rank"})
    lora_alpha: int = field(default=128, metadata={"help": "LoRA alpha"})
    lora_dropout: float = field(default=0.05, metadata={"help": "LoRA dropout"})
    target_modules: str = field(
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj,lm_head",
        metadata={"help": "Comma-separated list of target modules"}
    )


class GoCoderTrainer:
    """Main trainer class for Go Coder LoRA fine-tuning"""

    def __init__(self, config_path: str = "configs/training_config.yaml"):
        """Initialize trainer with configuration"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        # Set seed for reproducibility
        set_seed(42)

        # Initialize W&B if configured
        if os.getenv("WANDB_API_KEY"):
            self._init_wandb()

    def _init_wandb(self):
        """Initialize Weights & Biases tracking"""
        wandb.init(
            project=self.config['wandb']['project'],
            entity=self.config['wandb']['entity'],
            name=self.config['wandb']['name'],
            tags=self.config['wandb']['tags'],
            config=self.config
        )
        logger.info("Weights & Biases initialized")

    def load_model_and_tokenizer(self):
        """Load base model and tokenizer with quantization"""
        logger.info(f"Loading model: {self.config['model']['base_model']}")

        # Quantization configuration
        bnb_config = None
        if self.config['model']['load_in_4bit']:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type=self.config['model']['bnb_4bit_quant_type'],
                bnb_4bit_use_double_quant=self.config['model']['bnb_4bit_use_double_quant']
            )

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config['model']['base_model'],
            trust_remote_code=True,
            use_auth_token=os.getenv("HF_TOKEN")
        )

        # Set padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config['model']['base_model'],
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            use_auth_token=os.getenv("HF_TOKEN")
        )

        # Prepare model for k-bit training
        if self.config['model']['load_in_4bit']:
            self.model = prepare_model_for_kbit_training(self.model)

        logger.info("Model and tokenizer loaded successfully")

    def apply_lora(self):
        """Apply LoRA configuration to the model"""
        logger.info("Applying LoRA configuration")

        # Parse target modules
        target_modules = self.config['lora']['target_modules']

        # Create LoRA configuration
        lora_config = LoraConfig(
            r=self.config['lora']['r'],
            lora_alpha=self.config['lora']['lora_alpha'],
            lora_dropout=self.config['lora']['lora_dropout'],
            bias=self.config['lora']['bias'],
            task_type=TaskType.CAUSAL_LM,
            target_modules=target_modules
        )

        # Apply LoRA to model
        self.model = get_peft_model(self.model, lora_config)

        # Print trainable parameters
        self.model.print_trainable_parameters()

        logger.info("LoRA applied successfully")

    def load_datasets(self):
        """Load and prepare training and evaluation datasets"""
        logger.info("Loading datasets")

        # Load training dataset
        train_data = []
        with open(self.config['data']['train_dataset'], 'r') as f:
            for line in f:
                train_data.append(json.loads(line))

        # Load evaluation dataset
        eval_data = []
        with open(self.config['data']['eval_dataset'], 'r') as f:
            for line in f:
                eval_data.append(json.loads(line))

        # Format datasets
        self.train_dataset = self._format_dataset(train_data)
        self.eval_dataset = self._format_dataset(eval_data)

        logger.info(f"Loaded {len(self.train_dataset)} training examples")
        logger.info(f"Loaded {len(self.eval_dataset)} evaluation examples")

    def _format_dataset(self, data):
        """Format dataset for training"""
        formatted_data = []

        for item in data:
            # Apply prompt template
            prompt = self.config['data']['prompt_template'].format(
                instruction=item['instruction'],
                response=item['output']
            )

            # Tokenize
            tokenized = self.tokenizer(
                prompt,
                truncation=True,
                max_length=self.config['data']['max_length'],
                padding="max_length",
                return_tensors=None
            )

            # Add labels (same as input_ids for causal LM)
            tokenized['labels'] = tokenized['input_ids'].copy()

            formatted_data.append(tokenized)

        return Dataset.from_list(formatted_data)

    def create_training_arguments(self):
        """Create training arguments"""
        return TrainingArguments(
            output_dir=self.config['training']['output_dir'],
            num_train_epochs=self.config['training']['num_train_epochs'],
            per_device_train_batch_size=self.config['training']['per_device_train_batch_size'],
            per_device_eval_batch_size=self.config['training']['per_device_eval_batch_size'],
            gradient_accumulation_steps=self.config['training']['gradient_accumulation_steps'],
            gradient_checkpointing=self.config['training']['gradient_checkpointing'],
            learning_rate=self.config['training']['learning_rate'],
            lr_scheduler_type=self.config['training']['lr_scheduler_type'],
            warmup_ratio=self.config['training']['warmup_ratio'],
            optim=self.config['training']['optim'],
            weight_decay=self.config['training']['weight_decay'],
            max_grad_norm=self.config['training']['max_grad_norm'],
            fp16=self.config['training']['fp16'],
            bf16=self.config['training']['bf16'],
            logging_steps=self.config['training']['logging_steps'],
            save_steps=self.config['training']['save_steps'],
            eval_steps=self.config['training']['eval_steps'],
            save_total_limit=self.config['training']['save_total_limit'],
            load_best_model_at_end=self.config['training']['load_best_model_at_end'],
            metric_for_best_model=self.config['training']['metric_for_best_model'],
            greater_is_better=self.config['training']['greater_is_better'],
            push_to_hub=False,
            report_to=["wandb"] if os.getenv("WANDB_API_KEY") else ["tensorboard"],
            run_name=self.config['wandb']['name'],
            ddp_find_unused_parameters=False if self.config['training']['gradient_checkpointing'] else None,
        )

    def train(self):
        """Main training function"""
        logger.info("Starting training")

        # Load model and tokenizer
        self.load_model_and_tokenizer()

        # Apply LoRA
        self.apply_lora()

        # Load datasets
        self.load_datasets()

        # Create training arguments
        training_args = self.create_training_arguments()

        # Create data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )

        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            data_collator=data_collator,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )

        # Train
        trainer.train()

        # Save final model
        logger.info("Saving final model")
        trainer.save_model(os.path.join(self.config['training']['output_dir'], "final"))

        # Save tokenizer
        self.tokenizer.save_pretrained(os.path.join(self.config['training']['output_dir'], "final"))

        logger.info("Training complete!")

        # Push to hub if configured
        if os.getenv("HF_TOKEN"):
            self._push_to_hub(trainer)

    def _push_to_hub(self, trainer):
        """Push model to Hugging Face Hub"""
        try:
            model_name = f"go-coder-mistral-7b-lora"
            logger.info(f"Pushing model to Hugging Face Hub as {model_name}")

            trainer.push_to_hub(
                repo_name=model_name,
                use_auth_token=os.getenv("HF_TOKEN")
            )

            logger.info("Model pushed to Hub successfully")
        except Exception as e:
            logger.error(f"Failed to push to Hub: {e}")

    def evaluate(self, checkpoint_path: Optional[str] = None):
        """Evaluate the model on test set"""
        logger.info("Starting evaluation")

        if checkpoint_path:
            # Load specific checkpoint
            self.model = AutoModelForCausalLM.from_pretrained(
                checkpoint_path,
                device_map="auto",
                trust_remote_code=True
            )
            self.tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
        else:
            # Use current model
            if not hasattr(self, 'model'):
                self.load_model_and_tokenizer()
                self.apply_lora()

        # Load test dataset
        test_data = []
        with open(self.config['data']['test_dataset'], 'r') as f:
            for line in f:
                test_data.append(json.loads(line))

        # Evaluate
        results = []
        for item in tqdm(test_data[:10], desc="Evaluating"):  # Sample evaluation
            prompt = f"[INST] You are an expert Go programmer. {item['instruction']} [/INST]"

            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.95
                )

            generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            results.append({
                "instruction": item['instruction'],
                "expected": item['output'],
                "generated": generated.split("[/INST]")[-1].strip()
            })

        # Save evaluation results
        eval_dir = Path("./evaluation_results")
        eval_dir.mkdir(exist_ok=True)

        with open(eval_dir / "evaluation_results.json", 'w') as f:
            json.dump(results, f, indent=2)

        logger.info(f"Evaluation complete. Results saved to {eval_dir}")


def main():
    """Main execution function"""
    import argparse

    parser = argparse.ArgumentParser(description="Train LoRA for Go Coder")
    parser.add_argument("--config", type=str, default="configs/training_config.yaml",
                       help="Path to configuration file")
    parser.add_argument("--evaluate", action="store_true",
                       help="Run evaluation only")
    parser.add_argument("--checkpoint", type=str, default=None,
                       help="Path to checkpoint for evaluation")

    args = parser.parse_args()

    # Initialize trainer
    trainer = GoCoderTrainer(config_path=args.config)

    if args.evaluate:
        # Run evaluation
        trainer.evaluate(checkpoint_path=args.checkpoint)
    else:
        # Run training
        trainer.train()

        # Run evaluation after training
        trainer.evaluate()


if __name__ == "__main__":
    main()
