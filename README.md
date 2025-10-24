# Meta-Cognitive AI Agent

A modular AI agent framework with self-reflection capabilities and OpenAI GPT integration for real question answering.

## Overview

This project implements a sophisticated modular AI agent that can:
- Process and understand user queries through intent classification
- Use OpenAI's ChatGPT API to provide real, intelligent answers
- Maintain memory of past interactions
- Perform self-reflection and adapt strategies based on performance
- Handle replanning when confidence is low
- Log all interactions for transparency

## Architecture

The framework consists of several key components:

- **InputHandler**: Preprocesses raw input and classifies intent (general, explain, optimization, fetch)
- **CognitiveEngine**: Uses GPT to reason about queries and propose actions with confidence scores
- **ActionExecutor**: Executes actions by querying ChatGPT and returning real answers
- **FeedbackEvaluator**: Evaluates action outcomes and adjusts confidence scores
- **MemoryKB**: Stores past actions, results, and learned strategies
- **StrategyAdapter**: Adapts reasoning strategies based on performance
- **MetaCognitiveLayer**: Monitors system performance and triggers periodic self-reflection
- **ModularAgent**: Top-level orchestrator that coordinates all components

## Requirements

- Python 3.8+
- OpenAI API key

## Installation

1. Clone the repository:
```bash
git clone https://github.com/KendrickLamarr/meta-cognitive-ai.git
cd meta-cognitive-ai
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set your OpenAI API key:
```bash
export OPENAI_API_KEY='your-api-key-here'
```

## Usage

### Basic Usage

```python
from modular_agent_gpt import ModularAgent

# Initialize the agent
agent = ModularAgent(
    gpt_model="gpt-3.5-turbo",
    temperature=0.7,
    max_tokens=500
)

# Ask a question and get the answer
answer = agent.handle_query("What is machine learning?")
print(answer)
```

### Advanced Configuration

```python
agent = ModularAgent(
    replanning_threshold=0.6,      # Trigger replanning below this confidence
    reflection_frequency=5,         # Self-reflect every N queries
    failure_trigger=3,              # Self-reflect after N failures
    gpt_model="gpt-4",             # Use GPT-4 for better quality
    temperature=0.7,                # Sampling temperature (0.0-2.0)
    max_tokens=1000                 # Max tokens per response
)
```

### Running the Demo

Run the included demo with multiple example queries:

```bash
python modular_agent_gpt.py
```

### Running Tests

Test the agent structure without making API calls:

```bash
python test_agent.py
```

With an API key set, the test will also verify real API integration.

## Features

### Real GPT Integration
- Uses OpenAI's Chat Completion API for actual reasoning
- Configurable model selection (gpt-3.5-turbo, gpt-4, etc.)
- Adjustable temperature and token limits

### Intent Classification
- Automatically classifies queries into intents: general, explain, optimization, fetch
- Uses intent-specific system prompts for better responses

### Self-Reflection & Adaptation
- Monitors performance metrics (success rate, confidence)
- Adapts strategies when performance degrades
- Triggers replanning for low-confidence responses

### Memory & Learning
- Stores all queries, responses, and confidence scores
- Learns from past interactions
- Maintains strategy history

### Comprehensive Logging
- Logs all major operations at INFO level
- Debug logging available for detailed troubleshooting
- Tracks GPT API calls and responses

## File Structure

- `modular_agent.py` - Original framework with simulated actions
- `modular_agent_gpt.py` - Enhanced version with OpenAI GPT integration
- `test_agent.py` - Test suite with mock and real API tests
- `requirements.txt` - Python dependencies

## API Cost Considerations

The agent makes multiple API calls per query:
1. One call in CognitiveEngine for reasoning/confidence scoring
2. One call in ActionExecutor for the actual answer
3. Additional calls if replanning is triggered

To minimize costs:
- Use `gpt-3.5-turbo` instead of `gpt-4`
- Set lower `max_tokens` values
- Increase `replanning_threshold` to reduce replanning frequency

## Examples

### Example 1: Factual Question
```python
answer = agent.handle_query("What is the capital of France?")
# Returns: "The capital of France is Paris."
```

### Example 2: Explanation Request
```python
answer = agent.handle_query("Explain how neural networks work")
# Returns: Detailed explanation of neural networks
```

### Example 3: Optimization Advice
```python
answer = agent.handle_query("How can I optimize my Python code?")
# Returns: Practical optimization tips
```

## Accessing Memory and Statistics

```python
# View recent memory entries
recent = agent.memory.recent(5)

# Get average confidence
avg_confidence = agent.memory.average_confidence_recent(10)

# View learned strategies
strategies = agent.memory.get_strategies()

# Access full action log
all_actions = agent.memory.actions_log
```

## License

MIT

## Contributing

Contributions welcome! Please feel free to submit a Pull Request.