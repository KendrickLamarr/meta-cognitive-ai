# Quick Start Guide

Get up and running with the Meta-Cognitive AI Agent in 5 minutes!

## Installation

```bash
# Clone the repository
git clone https://github.com/KendrickLamarr/meta-cognitive-ai.git
cd meta-cognitive-ai

# Install dependencies
pip install -r requirements.txt

# Set your OpenAI API key
export OPENAI_API_KEY='your-api-key-here'
```

## Your First Query

```python
from modular_agent_gpt import ModularAgent

# Create an agent
agent = ModularAgent()

# Ask a question
answer = agent.handle_query("What is machine learning?")
print(answer)
```

## Run the Demo

```bash
python modular_agent_gpt.py
```

This will run 5 example queries and show:
- The ChatGPT answer for each
- Confidence scores
- Self-reflection in action
- Session statistics

## Example Output

```
======================================================================
Query 1: What is the capital of France?
======================================================================

ChatGPT Answer:
----------------------------------------------------------------------
The capital of France is Paris. Paris is not only the political capital
but also the cultural and economic center of France, known for its art,
fashion, gastronomy, and culture.
----------------------------------------------------------------------

Confidence: 0.856
Success: True
```

## What Happens Under the Hood

```
User Query: "What is machine learning?"
    ↓
InputHandler: Classifies intent as "explain"
    ↓
CognitiveEngine: Queries GPT for reasoning, proposes actions
    ↓
ActionExecutor: Calls ChatGPT API for the actual answer
    ↓
FeedbackEvaluator: Evaluates response quality
    ↓
MemoryKB: Stores interaction and confidence
    ↓
MetaCognitiveLayer: Checks if reflection is needed
    ↓
Returns: "Machine learning is a subset of artificial intelligence..."
```

## Customize Your Agent

```python
# Create a more conservative agent
careful_agent = ModularAgent(
    gpt_model="gpt-4",              # Better quality
    temperature=0.3,                 # More focused
    max_tokens=1000,                 # Longer responses
    replanning_threshold=0.8,        # Higher threshold
    reflection_frequency=3           # Reflect more often
)

# Create a faster, cheaper agent
fast_agent = ModularAgent(
    gpt_model="gpt-3.5-turbo",      # Faster
    temperature=0.7,                 # Balanced
    max_tokens=200,                  # Shorter responses
    replanning_threshold=0.5         # Lower threshold
)
```

## Inspect Agent Memory

```python
# View recent interactions
for entry in agent.memory.recent(3):
    print(f"Query: {entry['query']}")
    print(f"Confidence: {entry['confidence']:.3f}")
    print(f"Success: {entry['result'].success}")
    print()

# Get performance metrics
avg_confidence = agent.memory.average_confidence_recent(10)
print(f"Average confidence: {avg_confidence:.3f}")

# View learned strategies
strategies = agent.memory.get_strategies()
print(f"Strategies: {strategies}")
```

## Different Intent Types

The agent automatically classifies queries into intents:

```python
# Fetch Intent
agent.handle_query("Get the latest Python version")

# Explain Intent  
agent.handle_query("Explain how neural networks work")

# Optimization Intent
agent.handle_query("How can I optimize my database queries?")

# General Intent
agent.handle_query("Tell me about artificial intelligence")
```

Each intent uses a specialized system prompt for better responses!

## Run Tests

```bash
# Run all tests (works without API key)
python test_agent.py

# Validate implementation
python validate_implementation.py

# Run comprehensive examples
python example_usage.py
```

## Common Issues

### "No API key found"
Set the environment variable:
```bash
export OPENAI_API_KEY='sk-...'
```

### "Rate limit exceeded"
You're making too many requests. Wait a moment or:
- Use a lower `reflection_frequency`
- Increase `replanning_threshold` to reduce replanning

### "Insufficient quota"
Check your OpenAI account has available credits.

## Next Steps

1. **Read the README.md** for complete documentation
2. **Check IMPLEMENTATION_SUMMARY.md** for technical details
3. **Run example_usage.py** to see all features in action
4. **Customize the agent** for your specific use case

## Tips for Best Results

1. **Be specific** in your queries for better answers
2. **Use GPT-4** for complex questions (slower but better)
3. **Use GPT-3.5-turbo** for simple questions (faster and cheaper)
4. **Monitor confidence scores** to gauge answer quality
5. **Check memory** to understand agent learning

## Support

For issues or questions:
- Check the README.md
- Review the example_usage.py file
- Read the IMPLEMENTATION_SUMMARY.md

## Architecture

```
ModularAgent
├── InputHandler (intent classification)
├── CognitiveEngine (GPT reasoning)
├── ActionExecutor (GPT queries)
├── FeedbackEvaluator (quality assessment)
├── MemoryKB (interaction storage)
├── StrategyAdapter (learning)
└── MetaCognitiveLayer (self-reflection)
```

## Performance

- **Minimum latency**: ~2 seconds (2 API calls)
- **With replanning**: ~4-6 seconds (4-6 API calls)
- **API calls per query**: 2-6 depending on confidence
- **Cost per query**: ~$0.001-0.01 (varies by model and length)

## Ready to Start!

You now have everything you need to use the Meta-Cognitive AI Agent. Have fun! 🚀
