"""
Meta-Cognitive AI Agent with OpenAI GPT Integration

This module implements a modular AI agent framework with real GPT-based reasoning.
It maintains all existing features while integrating OpenAI's ChatGPT API for
answering user questions.
"""
from __future__ import annotations
import os
import logging
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import statistics
import copy

# OpenAI SDK
from openai import OpenAI

# Logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ModularAgent")

# Initialize OpenAI client (only if API key is available)
_api_key = os.environ.get("OPENAI_API_KEY")
if _api_key:
    client = OpenAI(api_key=_api_key)
else:
    # Create a placeholder that will raise an error if used
    client = None
    logger.warning("[OpenAI] No API key found in OPENAI_API_KEY environment variable")

# -------------------- Data Classes --------------------

@dataclass
class ProposedAction:
    name: str
    params: Dict[str, Any]
    confidence: float  # 0.0 - 1.0

@dataclass
class ActionResult:
    success: bool
    output: Any
    details: Dict[str, Any] = field(default_factory=dict)

# -------------------- Input Handler --------------------

class InputHandler:
    def preprocess(self, raw_input: str) -> Dict[str, Any]:
        logger.debug("[InputHandler] Raw input received: %s", raw_input)
        tokens = [t.strip(".,!?").lower() for t in raw_input.split()]
        intent = "general"
        if any(k in tokens for k in ("optimize", "improve", "tune")):
            intent = "optimization"
        elif any(k in tokens for k in ("explain", "describe", "why", "how")):
            intent = "explain"
        elif any(k in tokens for k in ("fetch", "get", "retrieve")):
            intent = "fetch"
        processed = {
            "original": raw_input,
            "tokens": tokens,
            "intent": intent,
            "length": len(raw_input),
        }
        logger.info("[InputHandler] Processed input: %s", processed)
        return processed

# -------------------- Memory / Knowledge --------------------

class MemoryKB:
    def __init__(self):
        self.actions_log: List[Dict[str, Any]] = []
        self.learned_strategies: Dict[str, Any] = {}
        logger.debug("[MemoryKB] Initialized empty memory.")

    def store_action(self, query: str, proposed: ProposedAction, result: ActionResult):
        entry = {
            "query": query,
            "action": copy.deepcopy(proposed),
            "result": copy.deepcopy(result),
            "confidence": proposed.confidence,
        }
        self.actions_log.append(entry)
        logger.info("[MemoryKB] Stored action: %s", entry)

    def recent(self, n: int = 10) -> List[Dict[str, Any]]:
        recent = self.actions_log[-n:]
        logger.debug("[MemoryKB] Retrieved %d recent entries.", len(recent))
        return recent

    def failures_in_recent(self, n: int = 10) -> int:
        count = sum(1 for e in self.recent(n) if not e["result"].success)
        logger.debug("[MemoryKB] Failures in recent %d: %d", n, count)
        return count

    def average_confidence_recent(self, n: int = 10) -> float:
        confs = [e["confidence"] for e in self.recent(n)]
        avg = statistics.mean(confs) if confs else 1.0
        logger.debug("[MemoryKB] Average confidence in recent %d: %f", n, avg)
        return avg

    def save_strategy(self, name: str, strategy: Dict[str, Any]):
        self.learned_strategies[name] = copy.deepcopy(strategy)
        logger.info("[MemoryKB] Saved strategy '%s': %s", name, strategy)

    def get_strategies(self) -> Dict[str, Any]:
        logger.debug("[MemoryKB] Retrieved strategies: %s", self.learned_strategies)
        return copy.deepcopy(self.learned_strategies)

# -------------------- OpenAI Query Helper --------------------

def query_openai(
    prompt: str,
    system_message: str = "You are a helpful assistant.",
    model: str = "gpt-3.5-turbo",
    temperature: float = 0.7,
    max_tokens: int = 500
) -> Tuple[str, bool]:
    """
    Query OpenAI ChatGPT API with proper error handling.
    
    Args:
        prompt: User query/prompt
        system_message: System context for the assistant
        model: GPT model to use
        temperature: Sampling temperature (0.0-2.0)
        max_tokens: Maximum tokens in response
        
    Returns:
        Tuple of (response_text, success_flag)
    """
    if client is None:
        error_msg = "Error: OpenAI client not initialized. Please set OPENAI_API_KEY environment variable."
        logger.error("[OpenAI] %s", error_msg)
        return error_msg, False
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens
        )
        answer = response.choices[0].message.content.strip()
        logger.info("[OpenAI] Query successful. Response length: %d chars", len(answer))
        return answer, True
    except Exception as e:
        logger.error("[OpenAI] Query failed: %s", e)
        return f"Error: Could not get response from OpenAI. {str(e)}", False

# -------------------- Cognitive Engine --------------------

class CognitiveEngine:
    """
    Performs reasoning using OpenAI GPT to propose actions.
    Uses ChatGPT to generate candidate actions based on user input and context.
    """
    
    def __init__(
        self,
        strategies: Optional[Dict[str, Any]] = None,
        reasoning_params: Optional[Dict[str, Any]] = None,
        gpt_model: str = "gpt-3.5-turbo",
        temperature: float = 0.7,
        max_tokens: int = 500
    ):
        self.strategies = strategies or {"default": {"aggressiveness": 0.5}}
        self.params = reasoning_params or {"max_proposals": 3}
        self.gpt_model = gpt_model
        self.temperature = temperature
        self.max_tokens = max_tokens
        logger.debug(
            "[CognitiveEngine] Initialized with strategies=%s params=%s model=%s",
            self.strategies, self.params, self.gpt_model
        )

    def propose_actions(self, processed_input: Dict[str, Any], context: Dict[str, Any]) -> List[ProposedAction]:
        """
        Use GPT to analyze input and generate multiple candidate actions.
        """
        logger.info("[CognitiveEngine] Proposing actions for input intent=%s", processed_input.get("intent"))
        proposals: List[ProposedAction] = []

        intent = processed_input.get("intent", "general")
        strat = self._select_strategy_for_intent(intent)
        
        # Calculate base confidence
        base_conf = 0.9 - (len(processed_input.get("tokens", [])) * 0.01)
        base_conf *= 1.0 - (0.2 * (1 - strat.get("aggressiveness", 0.5)))

        # Build context-aware prompt for GPT
        memory_summary = context.get("memory_summary", {})
        system_msg = self._build_system_message(intent, memory_summary)
        user_prompt = processed_input["original"]
        
        # Query GPT for reasoning scores
        external_scores = self._query_gpt_for_reasoning(user_prompt, system_msg, context)

        # Generate proposals with GPT-enhanced confidence
        for i in range(self.params.get("max_proposals", 3)):
            name = f"fetch_answer" if i == 0 else f"{intent}_action_{i+1}"
            params = {
                "attempt": i + 1,
                "strategy": strat,
                "query": processed_input["original"],
                "intent": intent
            }
            
            # Combine base confidence with GPT reasoning score
            ext = external_scores[i] if i < len(external_scores) else 0.5
            confidence = max(0.0, min(1.0, base_conf * (0.6 + 0.4 * ext) - i * 0.1))
            
            proposals.append(ProposedAction(name=name, params=params, confidence=confidence))
            logger.debug("[CognitiveEngine] Proposal %s: conf=%.3f", name, confidence)

        proposals.sort(key=lambda p: p.confidence, reverse=True)
        logger.info("[CognitiveEngine] Proposed %d actions", len(proposals))
        return proposals

    def update_strategies(self, strategies: Dict[str, Any]):
        """Update cognitive strategies based on meta-cognitive reflection."""
        logger.info("[CognitiveEngine] Updating strategies: %s", strategies)
        self.strategies.update(strategies)

    def _select_strategy_for_intent(self, intent: str) -> Dict[str, Any]:
        """Select appropriate strategy for the given intent."""
        strat = self.strategies.get(intent) or self.strategies.get("default", {"aggressiveness": 0.5})
        logger.debug("[CognitiveEngine] Selected strategy for intent '%s': %s", intent, strat)
        return strat

    def _build_system_message(self, intent: str, memory_summary: Dict[str, Any]) -> str:
        """Build context-aware system message for GPT."""
        base_msg = "You are a helpful AI assistant."
        
        if intent == "explain":
            base_msg = "You are an expert at explaining complex topics clearly and concisely."
        elif intent == "optimization":
            base_msg = "You are an optimization expert who provides actionable recommendations."
        elif intent == "fetch":
            base_msg = "You are a knowledgeable assistant who provides accurate information."
        
        # Add memory context if available
        if memory_summary.get("recent_failures", 0) > 2:
            base_msg += " Be extra careful and thorough in your response."
        
        return base_msg

    def _query_gpt_for_reasoning(
        self,
        prompt: str,
        system_msg: str,
        context: Dict[str, Any]
    ) -> List[float]:
        """
        Query GPT to get reasoning about the query and generate confidence scores.
        This simulates multiple reasoning paths.
        """
        # For efficiency, we generate a quick reasoning check
        reasoning_prompt = f"Analyze this query briefly: '{prompt}'. How confident are you in answering this (0-10 scale)?"
        
        response, success = query_openai(
            reasoning_prompt,
            system_message=system_msg,
            model=self.gpt_model,
            temperature=self.temperature,
            max_tokens=50
        )
        
        if success:
            # Try to extract a numeric score
            try:
                # Look for numbers in the response
                import re
                numbers = re.findall(r'\d+(?:\.\d+)?', response)
                if numbers:
                    score = float(numbers[0]) / 10.0  # Normalize to 0-1
                    score = max(0.0, min(1.0, score))
                else:
                    score = 0.8  # Default high confidence
            except:
                score = 0.8
        else:
            score = 0.5  # Medium confidence on error
        
        logger.debug("[CognitiveEngine] GPT reasoning score: %.3f", score)
        
        # Generate slight variations for multiple proposals
        scores = [score * (1.0 - i * 0.1) for i in range(self.params.get("max_proposals", 3))]
        return scores

# -------------------- Action Executor --------------------

class ActionExecutor:
    """
    Executes proposed actions. For 'fetch_answer' actions, queries ChatGPT.
    Returns real answers from the GPT API.
    """
    
    def __init__(
        self,
        gpt_model: str = "gpt-3.5-turbo",
        temperature: float = 0.7,
        max_tokens: int = 500
    ):
        self.gpt_model = gpt_model
        self.temperature = temperature
        self.max_tokens = max_tokens
        logger.debug(
            "[ActionExecutor] Initialized with model=%s temp=%.2f max_tokens=%d",
            self.gpt_model, self.temperature, self.max_tokens
        )
    
    def execute(self, action: ProposedAction) -> ActionResult:
        """
        Execute the proposed action. If it's a 'fetch_answer' action,
        query ChatGPT and return the actual response.
        """
        logger.info("[ActionExecutor] Executing action %s with params %s", action.name, action.params)
        
        # Check if this is a fetch_answer action
        if action.name == "fetch_answer" or "query" in action.params:
            return self._execute_gpt_query(action)
        else:
            # Fallback to simulated execution for other action types
            return self._execute_simulated(action)
    
    def _execute_gpt_query(self, action: ProposedAction) -> ActionResult:
        """Execute a real GPT query and return the answer."""
        query = action.params.get("query", "")
        intent = action.params.get("intent", "general")
        
        # Build system message based on intent
        system_messages = {
            "explain": "You are an expert at explaining complex topics clearly and concisely.",
            "optimization": "You are an optimization expert who provides actionable recommendations.",
            "fetch": "You are a knowledgeable assistant who provides accurate information.",
            "general": "You are a helpful AI assistant."
        }
        system_msg = system_messages.get(intent, system_messages["general"])
        
        # Query OpenAI
        answer, success = query_openai(
            prompt=query,
            system_message=system_msg,
            model=self.gpt_model,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        
        result = ActionResult(
            success=success,
            output=answer,
            details={
                "executed": True,
                "action_type": "gpt_query",
                "model": self.gpt_model,
                "intent": intent,
                "response_length": len(answer)
            }
        )
        
        logger.info(
            "[ActionExecutor] GPT query execution: success=%s, response_length=%d",
            result.success, len(answer)
        )
        return result
    
    def _execute_simulated(self, action: ProposedAction) -> ActionResult:
        """Fallback simulated execution for non-GPT actions."""
        success_chance = action.confidence
        success = random.random() < success_chance
        output = {
            "executed_action": action.name,
            "params": action.params,
            "simulated_success_probability": success_chance
        }
        result = ActionResult(
            success=success,
            output=output,
            details={"executed": True, "action_type": "simulated"}
        )
        logger.info("[ActionExecutor] Simulated execution result: success=%s", result.success)
        return result

# -------------------- Feedback Evaluator --------------------

class FeedbackEvaluator:
    """
    Evaluates action outcomes and adjusts confidence.
    Can optionally use GPT to evaluate response quality.
    """
    
    def __init__(self, replanning_threshold: float = 0.6):
        self.replanning_threshold = replanning_threshold
        logger.debug("[FeedbackEvaluator] Initialized with replanning_threshold=%f", replanning_threshold)

    def evaluate(self, action: ProposedAction, result: ActionResult) -> Tuple[float, bool]:
        """
        Evaluate the action result and adjust confidence.
        For GPT responses, consider response quality.
        """
        logger.info("[FeedbackEvaluator] Evaluating action %s (orig_conf=%.3f)", action.name, action.confidence)
        adjusted_confidence = action.confidence
        
        if result.success:
            # Boost confidence on success
            adjusted_confidence = min(1.0, action.confidence + 0.1)
            
            # Additional boost for GPT responses with good length
            if result.details.get("action_type") == "gpt_query":
                response_length = result.details.get("response_length", 0)
                if response_length > 50:  # Substantive answer
                    adjusted_confidence = min(1.0, adjusted_confidence + 0.05)
        else:
            # Reduce confidence on failure
            adjusted_confidence = max(0.0, action.confidence - 0.2)

        # Add some randomness for external evaluation simulation
        external_adjust = self._placeholder_external_evaluator(result)
        adjusted_confidence = max(0.0, min(1.0, adjusted_confidence * (1.0 + external_adjust)))

        needs_replan = adjusted_confidence < self.replanning_threshold
        logger.info(
            "[FeedbackEvaluator] Adjusted_confidence=%.3f, needs_replan=%s",
            adjusted_confidence, needs_replan
        )
        return adjusted_confidence, needs_replan

    def _placeholder_external_evaluator(self, result: ActionResult) -> float:
        """Simulate external evaluation with small random adjustment."""
        adj = (0.5 - random.random()) * 0.1
        logger.debug("[FeedbackEvaluator] External evaluator placeholder adj=%f", adj)
        return adj

# -------------------- Strategy Adapter --------------------

class StrategyAdapter:
    """
    Adapts strategies based on self-reflection and performance analysis.
    Adjusts GPT interaction strategies based on response quality.
    """
    
    def adapt(self, memory: MemoryKB) -> Dict[str, Any]:
        """
        Analyze recent performance and adapt strategies accordingly.
        """
        logger.info("[StrategyAdapter] Running self-reflection to adapt strategies.")
        recent = memory.recent(20)
        adapted: Dict[str, Any] = {}

        failures = sum(1 for e in recent if not e["result"].success)
        avg_conf = memory.average_confidence_recent(20)

        logger.debug("[StrategyAdapter] recent_failures=%d, avg_conf=%.3f", failures, avg_conf)

        # Adapt strategy based on performance
        if failures >= 3 or avg_conf < 0.5:
            # Poor performance - be more conservative
            adapted["default"] = {"aggressiveness": max(0.1, 0.4)}
            logger.info("[StrategyAdapter] Adapting to more conservative strategy: %s", adapted["default"])
        else:
            # Good performance - can be more aggressive
            adapted["default"] = {"aggressiveness": min(0.9, 0.6)}
            logger.info("[StrategyAdapter] Adapting to more aggressive strategy: %s", adapted["default"])

        memory.save_strategy("auto_adapted_default", adapted["default"])
        return adapted

# -------------------- Meta-Cognitive Layer --------------------

class MetaCognitiveLayer:
    """
    Monitors system performance and triggers self-reflection.
    Adjusts strategies based on GPT response quality and success rates.
    """
    
    def __init__(
        self,
        memory: MemoryKB,
        adapter: StrategyAdapter,
        reflection_frequency: int = 5,
        failure_trigger: int = 3
    ):
        self.memory = memory
        self.adapter = adapter
        self.reflection_frequency = reflection_frequency
        self.failure_trigger = failure_trigger
        self.tasks_processed = 0
        logger.debug(
            "[MetaCognitiveLayer] Initialized with freq=%d, failure_trigger=%d",
            reflection_frequency, failure_trigger
        )

    def observe_and_maybe_reflect(self) -> Optional[Dict[str, Any]]:
        """
        Monitor performance and trigger reflection when appropriate.
        """
        self.tasks_processed += 1
        logger.debug("[MetaCognitiveLayer] Task processed count=%d", self.tasks_processed)

        # Periodic reflection
        if self.tasks_processed % self.reflection_frequency == 0:
            logger.info("[MetaCognitiveLayer] Periodic self-reflection triggered.")
            return self._reflect()

        # Failure-triggered reflection
        recent_failures = self.memory.failures_in_recent(self.reflection_frequency)
        if recent_failures >= self.failure_trigger:
            logger.info(
                "[MetaCognitiveLayer] Conditional self-reflection triggered due to failures=%d",
                recent_failures
            )
            return self._reflect()

        logger.debug("[MetaCognitiveLayer] No reflection triggered.")
        return None

    def _reflect(self) -> Dict[str, Any]:
        """Perform self-reflection and strategy adaptation."""
        adapted = self.adapter.adapt(self.memory)
        logger.info("[MetaCognitiveLayer] Reflection produced adapted strategies: %s", adapted)
        return adapted

# -------------------- Modular Agent --------------------

class ModularAgent:
    """
    Top-level modular AI agent with full GPT integration.
    Coordinates all components to answer user questions using ChatGPT.
    """
    
    def __init__(
        self,
        replanning_threshold: float = 0.6,
        reflection_frequency: int = 5,
        failure_trigger: int = 3,
        gpt_model: str = "gpt-3.5-turbo",
        temperature: float = 0.7,
        max_tokens: int = 500
    ):
        """
        Initialize the modular agent with all components.
        
        Args:
            replanning_threshold: Confidence threshold below which replanning is triggered
            reflection_frequency: Number of tasks between periodic self-reflections
            failure_trigger: Number of failures that trigger immediate reflection
            gpt_model: OpenAI model to use (e.g., 'gpt-3.5-turbo', 'gpt-4')
            temperature: Sampling temperature for GPT (0.0-2.0)
            max_tokens: Maximum tokens in GPT responses
        """
        self.input_handler = InputHandler()
        self.memory = MemoryKB()
        self.cognitive = CognitiveEngine(
            gpt_model=gpt_model,
            temperature=temperature,
            max_tokens=max_tokens
        )
        self.executor = ActionExecutor(
            gpt_model=gpt_model,
            temperature=temperature,
            max_tokens=max_tokens
        )
        self.evaluator = FeedbackEvaluator(replanning_threshold=replanning_threshold)
        self.strategy_adapter = StrategyAdapter()
        self.meta = MetaCognitiveLayer(
            memory=self.memory,
            adapter=self.strategy_adapter,
            reflection_frequency=reflection_frequency,
            failure_trigger=failure_trigger
        )
        logger.info("[ModularAgent] Agent initialized with GPT model=%s", gpt_model)

    def handle_query(self, raw_query: str) -> str:
        """
        Handle a user query and return the ChatGPT answer.
        
        This method orchestrates the full pipeline:
        1. Input preprocessing
        2. Action proposal via CognitiveEngine (using GPT)
        3. Action execution via ActionExecutor (queries ChatGPT)
        4. Feedback evaluation
        5. Memory storage
        6. Meta-cognitive reflection
        
        Args:
            raw_query: User's question/query string
            
        Returns:
            ChatGPT's answer as a string
        """
        logger.info("\n[ModularAgent] Handling new query: %s", raw_query)
        
        # Step 1: Preprocess input
        processed = self.input_handler.preprocess(raw_query)
        context = {"memory_summary": self._summarize_memory_for_context()}

        # Step 2: Propose actions using GPT reasoning
        proposals = self.cognitive.propose_actions(processed, context)
        
        attempted = 0
        final_answer = None
        
        # Step 3: Execute and evaluate actions with potential replanning
        for attempt_limit in range(3):
            if not proposals:
                logger.warning("[ModularAgent] No proposals returned by CognitiveEngine.")
                break

            top = proposals[0]
            logger.info("[ModularAgent] Top proposal: %s (conf=%.3f)", top.name, top.confidence)

            # Execute action (gets ChatGPT answer)
            result = self.executor.execute(top)
            
            # Store the answer
            if result.success and isinstance(result.output, str):
                final_answer = result.output

            # Evaluate result
            adjusted_conf, needs_replan = self.evaluator.evaluate(top, result)

            # Store in memory
            top_adj = copy.deepcopy(top)
            top_adj.confidence = adjusted_conf
            self.memory.store_action(raw_query, top_adj, result)

            attempted += 1

            # Meta-cognitive reflection
            adapted = self.meta.observe_and_maybe_reflect()
            if adapted:
                self.cognitive.update_strategies(adapted)

            # Decide whether to replan
            if needs_replan and attempted < 3:
                logger.info("[ModularAgent] Confidence low (%.3f). Replanning...", adjusted_conf)
                self.cognitive.params["max_proposals"] = min(
                    5, self.cognitive.params.get("max_proposals", 3) + 1
                )
                context["last_failure"] = {
                    "attempt": attempted,
                    "confidence": adjusted_conf,
                    "result": result.output
                }
                proposals = self.cognitive.propose_actions(processed, context)
                continue
            else:
                logger.info(
                    "[ModularAgent] Accepting result. success=%s, adjusted_confidence=%.3f",
                    result.success, adjusted_conf
                )
                break
        
        # Return the answer
        if final_answer:
            return final_answer
        else:
            return "Unable to generate a response. Please try again."

    def _summarize_memory_for_context(self) -> Dict[str, Any]:
        """Create a summary of recent memory for context."""
        recent = self.memory.recent(5)
        failures = sum(1 for e in recent if not e["result"].success)
        avg_conf = self.memory.average_confidence_recent(5) if recent else 1.0
        summary = {
            "recent_failures": failures,
            "avg_confidence_recent": avg_conf,
            "strategies": self.memory.get_strategies()
        }
        logger.debug("[ModularAgent] Memory summary for context: %s", summary)
        return summary

# -------------------- Demo --------------------

if __name__ == "__main__":
    """
    Demo showcasing the GPT-enhanced modular agent.
    Requires OPENAI_API_KEY environment variable to be set.
    """
    import sys
    
    # Check for API key
    if not os.environ.get("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set.")
        print("Please set it with: export OPENAI_API_KEY='your-api-key-here'")
        sys.exit(1)
    
    # Set logging to INFO for demo
    logger.setLevel(logging.INFO)
    
    # Initialize agent with GPT-3.5-turbo (faster and cheaper for demo)
    agent = ModularAgent(
        replanning_threshold=0.6,
        reflection_frequency=3,
        failure_trigger=2,
        gpt_model="gpt-3.5-turbo",
        temperature=0.7,
        max_tokens=500
    )
    
    # Demo queries covering different intents
    queries = [
        "What is the capital of France?",
        "Explain how neural networks learn from data",
        "How can I optimize my Python code for better performance?",
        "What are the benefits of using microservices architecture?",
        "Describe the process of photosynthesis in plants",
    ]
    
    print("\n" + "="*70)
    print("Meta-Cognitive AI Agent with GPT Integration - Demo")
    print("="*70)
    
    for i, q in enumerate(queries, 1):
        print(f"\n{'='*70}")
        print(f"Query {i}: {q}")
        print(f"{'='*70}")
        
        answer = agent.handle_query(q)
        
        print(f"\n{'ChatGPT Answer:':^70}")
        print("-"*70)
        print(answer)
        print("-"*70)
        
        # Show confidence from memory
        if agent.memory.actions_log:
            last_entry = agent.memory.actions_log[-1]
            print(f"\nConfidence: {last_entry['confidence']:.3f}")
            print(f"Success: {last_entry['result'].success}")
    
    # Summary statistics
    print(f"\n{'='*70}")
    print("Session Summary")
    print(f"{'='*70}")
    print(f"Total queries processed: {len(agent.memory.actions_log)}")
    print(f"Successful responses: {sum(1 for e in agent.memory.actions_log if e['result'].success)}")
    print(f"Average confidence: {agent.memory.average_confidence_recent(len(agent.memory.actions_log)):.3f}")
    print(f"Learned strategies: {agent.memory.get_strategies()}")
    print("="*70)
