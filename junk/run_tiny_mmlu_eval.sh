#!/bin/bash

# Set up OpenAI API key from file
export OPENAI_API_KEY=$(cat key.openai.secret)

# Run the evaluation with inspect command
inspect eval tiny_mmlu.py --model openai/gpt-4o-mini

# Note: To limit the number of samples for testing, add --limit N
# Example: inspect eval tiny_mmlu.py --model openai/gpt-4o-mini --limit 5 