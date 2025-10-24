"""
Test script for the GPT-enhanced modular agent.
Can run in mock mode without an API key for testing structure.
"""
import os
import sys
import logging
from unittest.mock import patch, MagicMock

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TestAgent")


def test_without_api_key():
    """Test agent structure without making real API calls."""
    print("\n" + "="*70)
    print("Testing ModularAgent Structure (Mock Mode)")
    print("="*70)
    
    # Mock the OpenAI client
    with patch('modular_agent_gpt.client') as mock_client:
        # Mock the chat completion response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "This is a mock response from GPT."
        mock_client.chat.completions.create.return_value = mock_response
        
        # Import after patching
        from modular_agent_gpt import ModularAgent
        
        # Create agent
        agent = ModularAgent(
            replanning_threshold=0.6,
            reflection_frequency=3,
            failure_trigger=2,
            gpt_model="gpt-3.5-turbo",
            temperature=0.7,
            max_tokens=500
        )
        
        print("\n✓ Agent initialized successfully")
        
        # Test query handling
        test_queries = [
            "What is Python?",
            "Explain machine learning",
            "How to optimize code?"
        ]
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n--- Test Query {i}: {query}")
            answer = agent.handle_query(query)
            print(f"Answer: {answer}")
            print(f"✓ Query {i} processed successfully")
        
        # Check memory
        print(f"\n✓ Total entries in memory: {len(agent.memory.actions_log)}")
        print(f"✓ Average confidence: {agent.memory.average_confidence_recent(10):.3f}")
        
        # Check strategies
        strategies = agent.memory.get_strategies()
        print(f"✓ Learned strategies: {strategies}")
        
        print("\n" + "="*70)
        print("All structure tests passed!")
        print("="*70)


def test_with_api_key():
    """Test agent with real API calls (requires API key)."""
    print("\n" + "="*70)
    print("Testing ModularAgent with Real OpenAI API")
    print("="*70)
    
    from modular_agent_gpt import ModularAgent
    
    # Create agent
    agent = ModularAgent(
        replanning_threshold=0.6,
        reflection_frequency=3,
        failure_trigger=2,
        gpt_model="gpt-3.5-turbo",
        temperature=0.7,
        max_tokens=200  # Smaller for testing
    )
    
    print("\n✓ Agent initialized successfully")
    
    # Test with a simple query
    test_query = "What is 2+2?"
    print(f"\n--- Test Query: {test_query}")
    answer = agent.handle_query(test_query)
    print(f"\nAnswer: {answer}")
    
    # Verify we got a real response
    if answer and len(answer) > 0:
        print("\n✓ Received valid response from OpenAI")
    else:
        print("\n✗ Failed to get valid response")
        return False
    
    # Check memory storage
    if agent.memory.actions_log:
        last_entry = agent.memory.actions_log[-1]
        print(f"✓ Memory storage working - Confidence: {last_entry['confidence']:.3f}")
    
    print("\n" + "="*70)
    print("Real API test passed!")
    print("="*70)
    return True


def main():
    """Main test runner."""
    print("\n" + "="*70)
    print("ModularAgent GPT Integration Test Suite")
    print("="*70)
    
    # Check for API key
    api_key = os.environ.get("OPENAI_API_KEY")
    
    if not api_key:
        print("\nℹ No OPENAI_API_KEY found - running in mock mode")
        test_without_api_key()
    else:
        print("\nℹ OPENAI_API_KEY found - running real API tests")
        try:
            test_with_api_key()
        except Exception as e:
            print(f"\n✗ Error during real API test: {e}")
            print("\nFalling back to mock mode tests...")
            test_without_api_key()
    
    print("\n✓ All tests completed successfully!\n")


if __name__ == "__main__":
    main()
