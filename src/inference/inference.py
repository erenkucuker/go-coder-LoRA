"""
Inference Script for Go Coder LoRA Model
Provides easy-to-use interface for generating Go code with the fine-tuned Mistral 7B model
"""

import os
import sys
import logging
from typing import Optional, Dict, Any, List
from pathlib import Path
import json
import time

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TextStreamer
)
from peft import PeftModel
import yaml
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class GoCoderInference:
    """Main inference class for Go Coder LoRA model"""

    def __init__(
        self,
        model_path: str,
        config_path: str = "configs/training_config.yaml",
        load_in_4bit: bool = True,
        device: Optional[str] = None
    ):
        """Initialize the inference engine

        Args:
            model_path: Path to the fine-tuned LoRA model
            config_path: Path to configuration file
            load_in_4bit: Whether to load model in 4-bit quantization
            device: Device to use (cuda/cpu/auto)
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.model_path = model_path
        self.load_in_4bit = load_in_4bit

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.model = None
        self.tokenizer = None
        self.streamer = None

        logger.info(f"Inference engine initialized with model: {model_path}")
        logger.info(f"Using device: {self.device}")

    def load_model(self):
        """Load the fine-tuned model and tokenizer"""
        logger.info("Loading model and tokenizer...")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True
        )

        # Set padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # Setup text streamer for real-time output
        self.streamer = TextStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True
        )

        # Check if this is a LoRA model or full model
        adapter_config_path = Path(self.model_path) / "adapter_config.json"

        if adapter_config_path.exists():
            # This is a LoRA model
            logger.info("Loading LoRA model...")

            # Quantization config if requested
            bnb_config = None
            if self.load_in_4bit:
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True
                )

            # Load base model
            base_model = AutoModelForCausalLM.from_pretrained(
                self.config['model']['base_model'],
                quantization_config=bnb_config,
                device_map="auto" if self.device == "cuda" else None,
                torch_dtype=torch.float16,
                trust_remote_code=True,
                use_auth_token=os.getenv("HF_TOKEN")
            )

            # Apply LoRA
            self.model = PeftModel.from_pretrained(
                base_model,
                self.model_path,
                device_map="auto" if self.device == "cuda" else None
            )

            logger.info("LoRA model loaded successfully")
        else:
            # Load full model
            logger.info("Loading full model...")

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                device_map="auto" if self.device == "cuda" else None,
                torch_dtype=torch.float16,
                trust_remote_code=True
            )

            logger.info("Full model loaded successfully")

        # Set to evaluation mode
        self.model.eval()

    def generate(
        self,
        instruction: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.95,
        top_k: int = 50,
        do_sample: bool = True,
        stream: bool = False
    ) -> str:
        """Generate Go code based on instruction

        Args:
            instruction: The instruction/prompt for code generation
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            do_sample: Whether to use sampling
            stream: Whether to stream output

        Returns:
            Generated Go code
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Format prompt
        prompt = self._format_prompt(instruction)

        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.config['data']['max_length']
        )

        # Move to device
        if self.device == "cuda":
            inputs = {k: v.cuda() for k, v in inputs.items()}

        # Generate
        start_time = time.time()

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                streamer=self.streamer if stream else None
            )

        generation_time = time.time() - start_time

        # Decode
        generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract response
        response = self._extract_response(generated, prompt)

        # Log statistics
        num_tokens = len(outputs[0]) - len(inputs['input_ids'][0])
        tokens_per_second = num_tokens / generation_time

        logger.info(f"Generated {num_tokens} tokens in {generation_time:.2f}s ({tokens_per_second:.1f} tokens/s)")

        return response

    def _format_prompt(self, instruction: str) -> str:
        """Format instruction into model prompt"""
        template = self.config['data']['prompt_template']

        # For inference, we only need the instruction part
        prompt = template.split('[/INST]')[0] + '[/INST]'
        prompt = prompt.replace('{instruction}', instruction)

        return prompt

    def _extract_response(self, generated: str, prompt: str) -> str:
        """Extract the model's response from generated text"""
        # Remove the prompt from generated text
        if "[/INST]" in generated:
            response = generated.split("[/INST]")[-1].strip()
        else:
            response = generated[len(prompt):].strip()

        return response

    def batch_generate(
        self,
        instructions: List[str],
        **generation_kwargs
    ) -> List[str]:
        """Generate responses for multiple instructions

        Args:
            instructions: List of instructions
            **generation_kwargs: Generation parameters

        Returns:
            List of generated responses
        """
        responses = []

        for i, instruction in enumerate(instructions, 1):
            logger.info(f"Generating response {i}/{len(instructions)}")
            response = self.generate(instruction, **generation_kwargs)
            responses.append(response)

        return responses

    def interactive_mode(self):
        """Run interactive mode for testing"""
        print("\n" + "="*50)
        print("Go Coder LoRA - Interactive Mode")
        print("="*50)
        print("Type 'exit' to quit, 'clear' to clear screen")
        print("="*50 + "\n")

        while True:
            try:
                # Get user input
                instruction = input("\n💻 Enter your Go coding request:\n> ").strip()

                # Handle special commands
                if instruction.lower() == 'exit':
                    print("Goodbye!")
                    break
                elif instruction.lower() == 'clear':
                    os.system('clear' if os.name == 'posix' else 'cls')
                    continue
                elif not instruction:
                    continue

                # Generate response
                print("\n🤖 Generating Go code...\n")
                print("-" * 50)

                response = self.generate(
                    instruction,
                    stream=True,
                    temperature=0.7,
                    max_new_tokens=1024
                )

                print("-" * 50)

            except KeyboardInterrupt:
                print("\n\nInterrupted. Type 'exit' to quit.")
                continue
            except Exception as e:
                logger.error(f"Error during generation: {e}")
                print(f"\n❌ Error: {e}")


class GoCoderAPI:
    """API wrapper for Go Coder inference"""

    def __init__(self, model_path: str, **kwargs):
        """Initialize API with model"""
        self.inference_engine = GoCoderInference(model_path, **kwargs)
        self.inference_engine.load_model()

    def generate_function(
        self,
        description: str,
        function_name: Optional[str] = None,
        parameters: Optional[List[Dict[str, str]]] = None,
        return_type: Optional[str] = None
    ) -> str:
        """Generate a Go function based on specifications

        Args:
            description: Description of what the function should do
            function_name: Optional function name
            parameters: Optional list of parameter specs
            return_type: Optional return type specification

        Returns:
            Generated Go function code
        """
        # Build detailed instruction
        instruction = f"Write a Go function that {description}"

        if function_name:
            instruction += f"\nFunction name: {function_name}"

        if parameters:
            param_str = ", ".join([f"{p['name']} {p['type']}" for p in parameters])
            instruction += f"\nParameters: {param_str}"

        if return_type:
            instruction += f"\nReturn type: {return_type}"

        instruction += "\nProvide complete, production-ready code with proper error handling."

        return self.inference_engine.generate(instruction, max_new_tokens=512)

    def fix_code(self, code: str, error_message: str) -> str:
        """Fix Go code based on error message

        Args:
            code: The problematic Go code
            error_message: The error message

        Returns:
            Fixed Go code
        """
        instruction = f"""Fix the following Go code that has an error:

Code:
```go
{code}
```

Error:
{error_message}

Provide the corrected code with an explanation of what was wrong."""

        return self.inference_engine.generate(instruction, max_new_tokens=1024)

    def explain_code(self, code: str) -> str:
        """Explain Go code

        Args:
            code: Go code to explain

        Returns:
            Explanation of the code
        """
        instruction = f"""Explain the following Go code in detail:

```go
{code}
```

Include:
1. What the code does
2. How it works
3. Any potential issues or improvements
4. Best practices demonstrated or violated"""

        return self.inference_engine.generate(instruction, max_new_tokens=1024)

    def optimize_code(self, code: str) -> str:
        """Optimize Go code for performance

        Args:
            code: Go code to optimize

        Returns:
            Optimized Go code
        """
        instruction = f"""Optimize the following Go code for better performance:

```go
{code}
```

Provide:
1. The optimized version
2. Explanation of optimizations made
3. Performance implications"""

        return self.inference_engine.generate(instruction, max_new_tokens=1024)


def main():
    """Main execution function"""
    import argparse

    parser = argparse.ArgumentParser(description="Go Coder LoRA Inference")
    parser.add_argument("--model", type=str, required=True,
                       help="Path to the fine-tuned model")
    parser.add_argument("--instruction", type=str, default=None,
                       help="Single instruction to generate code for")
    parser.add_argument("--interactive", action="store_true",
                       help="Run in interactive mode")
    parser.add_argument("--batch", type=str, default=None,
                       help="Path to JSON file with batch instructions")
    parser.add_argument("--max-tokens", type=int, default=512,
                       help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7,
                       help="Sampling temperature")
    parser.add_argument("--no-4bit", action="store_true",
                       help="Disable 4-bit quantization")
    parser.add_argument("--output", type=str, default=None,
                       help="Output file for results")

    args = parser.parse_args()

    # Initialize inference engine
    engine = GoCoderInference(
        model_path=args.model,
        load_in_4bit=not args.no_4bit
    )
    engine.load_model()

    # Handle different modes
    if args.interactive:
        # Interactive mode
        engine.interactive_mode()

    elif args.batch:
        # Batch mode
        with open(args.batch, 'r') as f:
            batch_data = json.load(f)

        instructions = batch_data if isinstance(batch_data, list) else batch_data['instructions']

        print(f"Processing {len(instructions)} instructions...")
        responses = engine.batch_generate(
            instructions,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature
        )

        # Save results
        results = [
            {"instruction": inst, "response": resp}
            for inst, resp in zip(instructions, responses)
        ]

        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"Results saved to {args.output}")
        else:
            for i, result in enumerate(results, 1):
                print(f"\n{'='*50}")
                print(f"Example {i}:")
                print(f"Instruction: {result['instruction']}")
                print(f"Response:\n{result['response']}")

    elif args.instruction:
        # Single instruction mode
        response = engine.generate(
            args.instruction,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            stream=True
        )

        if args.output:
            with open(args.output, 'w') as f:
                f.write(response)
            print(f"\nOutput saved to {args.output}")

    else:
        print("Please specify either --instruction, --batch, or --interactive")
        sys.exit(1)


if __name__ == "__main__":
    main()
