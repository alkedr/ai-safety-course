from inspect_ai import Task, task
from inspect_ai.dataset import hf_dataset, Sample
from inspect_ai.scorer import match
from inspect_ai.solver import basic_agent, system_message
import logging

# Few-shot examples to help the model understand the task
FEW_SHOT_EXAMPLES = """
Example 1:
Question: What is the capital of France?
A. London
B. Berlin
C. Paris
D. Madrid

Answer: C

Example 2:
Question: Which planet is known as the Red Planet?
A. Venus
B. Mars
C. Jupiter
D. Saturn

Answer: B
"""

@task
def tiny_mmlu():
    """
    Evaluate a model on the tinyMMLU benchmark from HuggingFace.
    
    tinyMMLU is a concise version of the MMLU dataset, offering a subset of 100 data points
    selected from the original compilation. It's designed to efficiently estimate the performance
    of a large language model (LLM) with reduced dataset size, saving computational resources
    while maintaining the essence of the MMLU evaluation.
    
    The dataset contains multiple-choice questions from various domains of knowledge,
    including humanities, social sciences, STEM, and other areas.
    
    This evaluation:
    1. Loads the tinyMMLU dataset from HuggingFace
    2. Formats each question with its multiple-choice options
    3. Asks the model to select the correct answer (A, B, C, or D)
    4. Scores the model based on exact match with the correct answer
    
    Returns:
        Task: An inspect-ai Task object configured for the tinyMMLU benchmark
    """
    try:
        # Load the tinyMMLU dataset from HuggingFace
        dataset = hf_dataset(
            path="tinyBenchmarks/tinyMMLU",
            split="test",
            trust=True,
            sample_fields=lambda record: process_record(record)
        )
        
        logging.info(f"Successfully loaded tinyMMLU dataset with {len(dataset)} samples")
        
        return Task(
            dataset=dataset,
            solver=basic_agent(
                max_attempts=1,
                message_limit=10,
                system=system_message(
                    f"""You are an AI assistant taking a multiple-choice exam.
                    Read each question carefully and select the best answer from the given options.
                    Respond with just the letter of the correct answer (A, B, C, or D).
                    
                    Here are some examples of how to answer:
                    {FEW_SHOT_EXAMPLES}
                    """
                ),
            ),
            scorer=match(r"(?i)^\s*([A-D])\b.*$"),
        )
    except Exception as e:
        logging.error(f"Error loading tinyMMLU dataset: {e}")
        raise

def process_record(record):
    """
    Process a record from the dataset, validating its structure and formatting it.
    
    Args:
        record (dict): A record from the tinyMMLU dataset
        
    Returns:
        Sample: An inspect-ai Sample object with input and target fields
        
    Raises:
        ValueError: If the record is missing required fields or has invalid structure
    """
    # Validate record structure
    if "question" not in record:
        raise ValueError("Record missing 'question' field")
    if "choices" not in record:
        raise ValueError("Record missing 'choices' field")
    if "answer" not in record:
        raise ValueError("Record missing 'answer' field")
    
    # Validate choices
    choices = record["choices"]
    if not isinstance(choices, list) or len(choices) != 4:
        raise ValueError(f"Expected 'choices' to be a list of 4 items, got {choices}")
    
    # Convert numeric answer to letter (A, B, C, D)
    answer = record["answer"]
    if isinstance(answer, int) and 0 <= answer <= 3:
        # Convert 0-3 to A-D
        answer_letter = chr(65 + answer)  # A, B, C, D
    elif isinstance(answer, str) and answer in "ABCD":
        answer_letter = answer
    else:
        # Try to convert string to int
        try:
            answer_num = int(answer)
            if 0 <= answer_num <= 3:
                answer_letter = chr(65 + answer_num)
            else:
                raise ValueError(f"Expected 'answer' to be between 0-3 or A-D, got {answer}")
        except (ValueError, TypeError):
            raise ValueError(f"Expected 'answer' to be between 0-3 or A-D, got {answer}")
    
    # Format the question and return a Sample
    return Sample(
        input=format_question(record),
        target=answer_letter
    )

def format_question(record):
    """
    Format the question with choices for the model.
    
    Args:
        record (dict): A record from the tinyMMLU dataset containing 'question' and 'choices'
        
    Returns:
        str: Formatted question with lettered choices
    """
    question = record["question"]
    choices = record["choices"]
    
    formatted_question = f"Question: {question}\n\n"
    
    # Add choices with letters
    for i, choice in enumerate(choices):
        letter = chr(65 + i)  # A, B, C, D
        formatted_question += f"{letter}. {choice}\n"
    
    return formatted_question 