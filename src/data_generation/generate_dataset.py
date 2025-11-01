"""
Dataset Generation Pipeline using Claude API
Generates high-quality Go coding instruction-response pairs for LoRA fine-tuning
"""

import os
import json
import asyncio
import aiohttp
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict
import random
import hashlib

import anthropic
from anthropic import AsyncAnthropic
from dotenv import load_dotenv
import jsonlines
import yaml
from tqdm.asyncio import tqdm

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data_generation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class GoExample:
    """Represents a single Go coding example"""
    id: str
    instruction: str
    response: str
    topic: str
    difficulty: str
    metadata: Dict[str, Any]
    timestamp: str


class GoPromptGenerator:
    """Generates diverse Go programming prompts"""

    def __init__(self, config_path: str = "configs/training_config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.topics = self.config['generation']['topics']
        self.difficulties = ['beginner', 'intermediate', 'advanced', 'expert']

        # Go-specific prompt templates
        self.prompt_templates = {
            'implementation': [
                "Write a Go function that {task}",
                "Implement a Go solution for {task}",
                "Create a Go program that {task}",
                "Develop a Go package for {task}",
            ],
            'explanation': [
                "Explain how to {concept} in Go with a code example",
                "Demonstrate {concept} in Go with best practices",
                "Show how to properly implement {concept} in Go",
                "Provide a comprehensive example of {concept} in Go",
            ],
            'debugging': [
                "Debug and fix this Go code: {code}",
                "Identify and correct the issues in this Go function: {code}",
                "Refactor this Go code for better performance: {code}",
                "Improve the error handling in this Go code: {code}",
            ],
            'review': [
                "Review this Go code and suggest improvements: {code}",
                "Analyze this Go implementation for best practices: {code}",
                "Optimize this Go solution: {code}",
                "Convert this code to idiomatic Go: {code}",
            ],
            'design': [
                "Design a Go interface for {system}",
                "Create a concurrent Go solution for {problem}",
                "Architect a Go microservice for {service}",
                "Design a Go package structure for {project}",
            ]
        }

        # Topic-specific tasks
        self.topic_tasks = {
            "Go fundamentals and syntax": [
                "variable declaration and initialization",
                "working with slices and maps",
                "struct composition and methods",
                "type assertions and conversions",
                "defer statements and panic recovery",
            ],
            "Concurrency patterns (goroutines, channels)": [
                "worker pool pattern",
                "fan-in/fan-out pattern",
                "pipeline processing",
                "select statement with timeout",
                "mutex and atomic operations",
                "context cancellation",
            ],
            "Error handling best practices": [
                "custom error types",
                "error wrapping and unwrapping",
                "sentinel errors",
                "error handling in concurrent code",
                "graceful degradation",
            ],
            "Interface design and composition": [
                "interface segregation",
                "embedding interfaces",
                "type switches",
                "interface satisfaction",
                "mock implementations",
            ],
            "Testing and benchmarking": [
                "table-driven tests",
                "test fixtures and helpers",
                "benchmark functions",
                "fuzz testing",
                "test coverage analysis",
            ],
            "Web development with Go": [
                "HTTP server implementation",
                "middleware pattern",
                "RESTful API design",
                "WebSocket handling",
                "template rendering",
            ],
            "Database interactions": [
                "SQL query builders",
                "connection pooling",
                "transaction management",
                "migration handling",
                "ORM usage",
            ],
            "Performance optimization": [
                "memory profiling",
                "CPU profiling",
                "escape analysis",
                "string concatenation optimization",
                "slice pre-allocation",
            ],
        }

    def generate_prompt(self, topic: str, difficulty: str) -> str:
        """Generate a diverse prompt for the given topic and difficulty"""
        prompt_type = random.choice(list(self.prompt_templates.keys()))

        if prompt_type in ['implementation', 'explanation', 'design']:
            # Get task/concept specific to the topic
            if topic in self.topic_tasks:
                task = random.choice(self.topic_tasks[topic])
            else:
                task = f"a solution related to {topic}"

            template = random.choice(self.prompt_templates[prompt_type])

            if '{task}' in template:
                prompt = template.format(task=task)
            elif '{concept}' in template:
                prompt = template.format(concept=task)
            elif '{system}' in template or '{problem}' in template or '{service}' in template or '{project}' in template:
                prompt = template.format(
                    system=task, problem=task, service=task, project=task
                )
            else:
                prompt = template
        else:
            # For debugging and review, generate sample code
            prompt = self._generate_code_review_prompt(topic, difficulty)

        # Add difficulty context
        if difficulty == 'expert':
            prompt += ". Include advanced Go patterns, optimization techniques, and production-ready considerations."
        elif difficulty == 'advanced':
            prompt += ". Use appropriate Go idioms and consider performance implications."
        elif difficulty == 'beginner':
            prompt += ". Provide clear explanations and comments for learning purposes."

        return prompt

    def _generate_code_review_prompt(self, topic: str, difficulty: str) -> str:
        """Generate a code review/debugging prompt with sample code"""
        # Generate problematic code snippets for review
        code_samples = {
            "Concurrency patterns (goroutines, channels)": """
func processItems(items []string) {
    for _, item := range items {
        go func() {
            fmt.Println(item)
        }()
    }
}""",
            "Error handling best practices": """
func readFile(path string) string {
    data, _ := os.ReadFile(path)
    return string(data)
}""",
            "Interface design and composition": """
type Database interface {
    Connect() error
    Query(sql string) ([]map[string]interface{}, error)
    Insert(table string, data map[string]interface{}) error
    Update(table string, id int, data map[string]interface{}) error
    Delete(table string, id int) error
    Close() error
}""",
        }

        if topic in code_samples:
            code = code_samples[topic]
        else:
            code = """
func calculate(x, y int) int {
    result := x + y
    return result
}"""

        template = random.choice(self.prompt_templates['review'])
        return template.format(code=code)


class ClaudeDataGenerator:
    """Generates Go coding data using Claude API"""

    def __init__(self):
        self.api_key = os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY not found in environment variables")

        self.client = AsyncAnthropic(api_key=self.api_key)
        self.prompt_generator = GoPromptGenerator()

        # Load config
        with open("configs/training_config.yaml", 'r') as f:
            self.config = yaml.safe_load(f)

        # Create data directories
        self.data_dir = Path(self.config['data']['train_dataset']).parent
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Rate limiting
        self.semaphore = asyncio.Semaphore(5)  # Max 5 concurrent requests
        self.request_delay = 1.0  # Delay requests in seconds

    async def generate_single_example(
        self,
        topic: str,
        difficulty: str,
        example_num: int
    ) -> Optional[GoExample]:
        """Generate a single Go coding example using Claude"""
        async with self.semaphore:
            try:
                # Generate prompt
                instruction = self.prompt_generator.generate_prompt(topic, difficulty)

                # Create system prompt for knowledge distillation
                system_prompt = """You are an expert Go programmer and teacher.
                Provide high-quality, production-ready Go code with best practices.
                Include detailed explanations and follow Go idioms and conventions.
                Your responses should be educational and suitable for training other models.
                Always provide complete, runnable code examples when applicable."""

                # Call Claude API
                message = await self.client.messages.create(
                    model=self.config['generation']['claude_model'],
                    max_tokens=self.config['generation']['max_tokens'],
                    temperature=self.config['generation']['temperature'],
                    system=system_prompt,
                    messages=[
                        {"role": "user", "content": instruction}
                    ]
                )

                response = message.content[0].text

                # Validate response
                if self.config['generation']['validate_code']:
                    if not self._validate_go_code(response):
                        logger.warning(f"Invalid Go code generated for {topic} - {difficulty}")
                        return None

                # Create example object
                example_id = hashlib.md5(
                    f"{topic}_{difficulty}_{example_num}_{datetime.now().isoformat()}".encode()
                ).hexdigest()

                example = GoExample(
                    id=example_id,
                    instruction=instruction,
                    response=response,
                    topic=topic,
                    difficulty=difficulty,
                    metadata={
                        "model": self.config['generation']['claude_model'],
                        "temperature": self.config['generation']['temperature'],
                        "example_num": example_num,
                    },
                    timestamp=datetime.now().isoformat()
                )

                logger.info(f"Generated example {example_num} for {topic} ({difficulty})")

                # Rate limiting delay
                await asyncio.sleep(self.request_delay)

                return example

            except Exception as e:
                logger.error(f"Error generating example: {e}")
                return None

    def _validate_go_code(self, response: str) -> bool:
        """Basic validation of Go code in response"""
        # Check for common Go keywords and patterns
        go_indicators = [
            'func ', 'package ', 'import ', 'type ', 'var ', 'const ',
            'if ', 'for ', 'range ', 'return ', 'go ', 'chan ', 'select '
        ]

        # Check if response contains code blocks
        has_code = '```' in response or any(indicator in response for indicator in go_indicators)

        # Check minimum length
        min_length = self.config['generation'].get('min_code_length', 50)

        return has_code and len(response) >= min_length

    async def generate_dataset(self, num_examples_per_topic: Optional[int] = None):
        """Generate complete dataset for all topics"""
        if num_examples_per_topic is None:
            num_examples_per_topic = self.config['generation']['examples_per_topic']

        topics = self.config['generation']['topics']
        difficulties = ['beginner', 'intermediate', 'advanced', 'expert']

        all_examples = []
        total_tasks = len(topics) * num_examples_per_topic

        with tqdm(total=total_tasks, desc="Generating examples") as pbar:
            for topic in topics:
                topic_examples = []

                # Generate examples for each difficulty level
                for i in range(num_examples_per_topic):
                    difficulty = random.choice(difficulties)
                    example = await self.generate_single_example(topic, difficulty, i)

                    if example:
                        topic_examples.append(example)

                    pbar.update(1)

                all_examples.extend(topic_examples)

                # Save intermediate results
                self._save_examples(topic_examples, f"{topic.replace(' ', '_')}_examples.jsonl")

        # Split into train/eval/test sets
        self._split_and_save_dataset(all_examples)

        logger.info(f"Generated {len(all_examples)} total examples")
        return all_examples

    def _save_examples(self, examples: List[GoExample], filename: str):
        """Save examples to JSONL file"""
        output_path = self.data_dir / "raw" / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with jsonlines.open(output_path, mode='w') as writer:
            for example in examples:
                writer.write(asdict(example))

        logger.info(f"Saved {len(examples)} examples to {output_path}")

    def _split_and_save_dataset(self, examples: List[GoExample]):
        """Split dataset into train/eval/test sets and save"""
        random.shuffle(examples)

        total = len(examples)
        train_size = int(total * self.config['data']['train_split'])
        eval_size = int(total * self.config['data']['eval_split'])

        train_examples = examples[:train_size]
        eval_examples = examples[train_size:train_size + eval_size]
        test_examples = examples[train_size + eval_size:]

        # Save processed datasets
        datasets = {
            'train': train_examples,
            'eval': eval_examples,
            'test': test_examples
        }

        for split_name, split_examples in datasets.items():
            output_path = Path(self.config['data'][f'{split_name}_dataset'])
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with jsonlines.open(output_path, mode='w') as writer:
                for example in split_examples:
                    # Format for training
                    formatted = {
                        "instruction": example.instruction,
                        "input": "",
                        "output": example.response,
                        "metadata": {
                            "topic": example.topic,
                            "difficulty": example.difficulty,
                            **example.metadata
                        }
                    }
                    writer.write(formatted)

            logger.info(f"Saved {len(split_examples)} {split_name} examples to {output_path}")


async def main():
    """Main execution function"""
    generator = ClaudeDataGenerator()

    # Generate dataset
    examples = await generator.generate_dataset(num_examples_per_topic=10)  # Start with 10 per topic for testing

    print(f"\n✅ Dataset generation complete!")
    print(f"Total examples generated: {len(examples)}")
    print(f"Data saved to: {generator.data_dir}")


if __name__ == "__main__":
    asyncio.run(main())
