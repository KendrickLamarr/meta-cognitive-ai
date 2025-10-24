"""
Example usage script for the Meta-Cognitive AI Agent with GPT integration.

This demonstrates various ways to use the agent and access its features.
"""
import os
import sys
from modular_agent_gpt import ModularAgent


def example_basic_usage():
    """Example 1: Basic question answering"""
    print("\n" + "="*70)
    print("Example 1: Basic Question Answering")
    print("="*70)
    
    agent = ModularAgent()
    
    questions = [
        "What is Python?",
        "How does recursion work?",
        "What are the benefits of cloud computing?"
    ]
    
    for question in questions:
        print(f"\nQ: {question}")
        answer = agent.handle_query(question)
        print(f"A: {answer}\n")


def example_advanced_configuration():
    """Example 2: Using advanced configuration options"""
    print("\n" + "="*70)
    print("Example 2: Advanced Configuration")
    print("="*70)
    
    # Create agent with custom settings
    agent = ModularAgent(
        replanning_threshold=0.7,      # Higher threshold = more likely to replan
        reflection_frequency=2,         # Reflect more often
        failure_trigger=1,              # Reflect after just 1 failure
        gpt_model="gpt-3.5-turbo",     # Fast and cost-effective
        temperature=0.3,                # More focused/deterministic responses
        max_tokens=200                  # Shorter responses
    )
    
    question = "Explain quantum computing in simple terms"
    print(f"\nQ: {question}")
    answer = agent.handle_query(question)
    print(f"A: {answer}")
    
    # Show configuration impact
    print(f"\nConfiguration used:")
    print(f"  - Model: gpt-3.5-turbo")
    print(f"  - Temperature: 0.3 (focused)")
    print(f"  - Max tokens: 200 (concise)")


def example_memory_inspection():
    """Example 3: Inspecting memory and learned strategies"""
    print("\n" + "="*70)
    print("Example 3: Memory Inspection and Learning")
    print("="*70)
    
    agent = ModularAgent()
    
    # Process multiple queries
    queries = [
        "What is machine learning?",
        "How does supervised learning differ from unsupervised learning?",
        "What is deep learning?",
        "Explain reinforcement learning",
        "What are neural networks?"
    ]
    
    for query in queries:
        print(f"\nProcessing: {query}")
        agent.handle_query(query)
    
    # Inspect memory
    print("\n" + "-"*70)
    print("Memory Statistics:")
    print("-"*70)
    
    total_entries = len(agent.memory.actions_log)
    print(f"Total queries processed: {total_entries}")
    
    avg_confidence = agent.memory.average_confidence_recent(total_entries)
    print(f"Average confidence: {avg_confidence:.3f}")
    
    failures = agent.memory.failures_in_recent(total_entries)
    print(f"Total failures: {failures}")
    
    strategies = agent.memory.get_strategies()
    print(f"Learned strategies: {strategies}")
    
    # Show recent queries
    print("\n" + "-"*70)
    print("Recent Query Summary:")
    print("-"*70)
    for i, entry in enumerate(agent.memory.recent(3), 1):
        print(f"\n{i}. Query: {entry['query']}")
        print(f"   Action: {entry['action'].name}")
        print(f"   Confidence: {entry['confidence']:.3f}")
        print(f"   Success: {entry['result'].success}")


def example_different_intents():
    """Example 4: Testing different intent classifications"""
    print("\n" + "="*70)
    print("Example 4: Different Intent Types")
    print("="*70)
    
    agent = ModularAgent()
    
    # Queries with different intents
    test_cases = [
        ("fetch", "Get the latest Python version"),
        ("explain", "Explain how HTTP works"),
        ("optimization", "How can I improve database query performance?"),
        ("general", "Tell me about artificial intelligence")
    ]
    
    for expected_intent, query in test_cases:
        print(f"\n{'='*70}")
        print(f"Expected Intent: {expected_intent.upper()}")
        print(f"Query: {query}")
        print("-"*70)
        
        answer = agent.handle_query(query)
        print(f"Answer: {answer[:200]}...")  # Truncate for display
        
        # Check what intent was detected
        if agent.memory.actions_log:
            last_action = agent.memory.actions_log[-1]
            detected_intent = last_action['action'].params.get('intent', 'unknown')
            print(f"\nDetected Intent: {detected_intent.upper()}")
            print(f"Match: {'✓' if detected_intent == expected_intent else '✗'}")


def example_self_reflection():
    """Example 5: Observing self-reflection and adaptation"""
    print("\n" + "="*70)
    print("Example 5: Self-Reflection and Strategy Adaptation")
    print("="*70)
    
    # Create agent with frequent reflection
    agent = ModularAgent(
        reflection_frequency=3,  # Reflect every 3 queries
        failure_trigger=2
    )
    
    queries = [
        "What is Python?",
        "What is Java?",
        "What is JavaScript?",  # This will trigger reflection
        "What is C++?",
        "What is Ruby?",
        "What is Go?"  # This will trigger another reflection
    ]
    
    for i, query in enumerate(queries, 1):
        print(f"\n{'='*70}")
        print(f"Query {i}: {query}")
        print("-"*70)
        
        # Get current strategies before query
        strategies_before = agent.memory.get_strategies()
        
        answer = agent.handle_query(query)
        print(f"Answer: {answer[:100]}...")
        
        # Check if strategies changed (reflection occurred)
        strategies_after = agent.memory.get_strategies()
        
        if strategies_before != strategies_after:
            print("\n🔄 Self-Reflection Triggered!")
            print(f"New Strategy: {strategies_after}")


def main():
    """Run all examples"""
    print("\n" + "="*70)
    print("Meta-Cognitive AI Agent - Example Usage Guide")
    print("="*70)
    
    # Check for API key
    if not os.environ.get("OPENAI_API_KEY"):
        print("\n⚠ Error: OPENAI_API_KEY environment variable not set.")
        print("Please set it with: export OPENAI_API_KEY='your-api-key-here'")
        sys.exit(1)
    
    print("\nℹ Running examples with real OpenAI API calls...")
    print("Note: These examples will consume API tokens.\n")
    
    # Run examples
    try:
        # Comment out any examples you don't want to run
        example_basic_usage()
        example_advanced_configuration()
        example_memory_inspection()
        example_different_intents()
        example_self_reflection()
        
        print("\n" + "="*70)
        print("All examples completed successfully!")
        print("="*70 + "\n")
        
    except Exception as e:
        print(f"\n✗ Error during examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
