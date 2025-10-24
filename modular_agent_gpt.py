# modular_agent_openai.py
import os
import logging
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import statistics
import copy

# OpenAI SDK
from openai import OpenAI

# Logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ModularAgent")

# Initialize OpenAI client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

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

def query_openai(prompt: str, instructions: str = "You are a helpful coding assistant.") -> str:
    try:
        response = client.responses.create(
            model="gpt-4o",
            instructions=instructions,
            input=prompt
        )
        return response.output_text
    except Exception as e:
        logger.error("OpenAI query failed: %s", e)
        return "Error: Could not get response from OpenAI."

# -------------------- Cognitive Engine --------------------

class CognitiveEngine:
    def __init__(self, strategies: Optional[Dict[str, Any]] = None, reasoning_params: Optional[Dict[str, Any]] = None):
        self.strategies = strategies or {"default": {"aggressiveness": 0.5}}
        self.params = reasoning_params or {"max_proposals": 3}
        logger.debug("[CognitiveEngine] Initialized with strategies=%s params=%s", self.strategies, self.params)

    def propose_actions(self, processed_input: Dict[str, Any], context: Dict[str, Any]) -> List[ProposedAction]:
        logger.info("[CognitiveEngine] Proposing actions for input intent=%s", processed_input.get("intent"))
        proposals: List[ProposedAction] = []

        intent = processed_input.get("intent", "general")
        strat = self._select_strategy_for_intent(intent)
        base_conf = 0.9 - (len(processed_input.get("tokens", [])) * 0.01)
        base_conf *= 1.0 - (0.2 * (1 - strat.get("aggressiveness", 0.5)))

        # Get actual AI suggestion using OpenAI
        ai_text = query_openai(processed_input["original"], instructions=f"You are a helpful assistant for {intent} tasks.")
        logger.debug("[CognitiveEngine] OpenAI suggested text: %s", ai_text[:100])

        # Generate proposals
        for i in range(self.params.get("max_proposals", 3)):
            name = f"{intent}_action_{i+1}"
            params = {"attempt": i + 1, "strategy": strat, "ai_suggestion": ai_text}
            confidence = max(0.0, min(1.0, base_conf - i*0.1))
            proposals.append(ProposedAction(name=name, params=params, confidence=confidence))
            logger.debug("[CognitiveEngine] Proposal %s: conf=%.3f", name, confidence)

        proposals.sort(key=lambda p: p.confidence, reverse=True)
        return proposals

    def update_strategies(self, strategies: Dict[str, Any]):
        self.strategies.update(strategies)
        logger.info("[CognitiveEngine] Updated strategies: %s", strategies)

    def _select_strategy_for_intent(self, intent: str) -> Dict[str, Any]:
        return self.strategies.get(intent) or self.strategies.get("default", {"aggressiveness": 0.5})

# -------------------- Action Executor --------------------

class ActionExecutor:
    def execute(self, action: ProposedAction) -> ActionResult:
        logger.info("[ActionExecutor] Executing action %s", action.name)
        success = random.random() < action.confidence
        output = {"executed_action": action.name, "params": action.params, "simulated_success_probability": action.confidence}
        return ActionResult(success=success, output=output, details={"executed": True})

# -------------------- Feedback Evaluator --------------------

class FeedbackEvaluator:
    def __init__(self, replanning_threshold: float = 0.6):
        self.replanning_threshold = replanning_threshold

    def evaluate(self, action: ProposedAction, result: ActionResult):
        adjusted_conf = action.confidence
        if result.success:
            adjusted_conf = min(1.0, adjusted_conf + 0.1)
        else:
            adjusted_conf = max(0.0, adjusted_conf - 0.2)
        needs_replan = adjusted_conf < self.replanning_threshold
        return adjusted_conf, needs_replan

# -------------------- Strategy Adapter --------------------

class StrategyAdapter:
    def adapt(self, memory: MemoryKB) -> Dict[str, Any]:
        recent = memory.recent(20)
        failures = sum(1 for e in recent if not e["result"].success)
        avg_conf = memory.average_confidence_recent(20)
        adapted: Dict[str, Any] = {}
        if failures >= 3 or avg_conf < 0.5:
            adapted["default"] = {"aggressiveness": max(0.1, 0.4)}
        else:
            adapted["default"] = {"aggressiveness": min(0.9, 0.6)}
        memory.save_strategy("auto_adapted_default", adapted["default"])
        return adapted

# -------------------- Meta-Cognitive Layer --------------------

class MetaCognitiveLayer:
    def __init__(self, memory: MemoryKB, adapter: StrategyAdapter, reflection_frequency: int = 5, failure_trigger: int = 3):
        self.memory = memory
        self.adapter = adapter
        self.reflection_frequency = reflection_frequency
        self.failure_trigger = failure_trigger
        self.tasks_processed = 0

    def observe_and_maybe_reflect(self) -> Optional[Dict[str, Any]]:
        self.tasks_processed += 1
        if self.tasks_processed % self.reflection_frequency == 0 or self.memory.failures_in_recent(self.reflection_frequency) >= self.failure_trigger:
            return self.adapter.adapt(self.memory)
        return None

# -------------------- Modular Agent --------------------

class ModularAgent:
    def __init__(self, replanning_threshold=0.6, reflection_frequency=5, failure_trigger=3):
        self.input_handler = InputHandler()
        self.memory = MemoryKB()
        self.cognitive = CognitiveEngine()
        self.executor = ActionExecutor()
        self.evaluator = FeedbackEvaluator(replanning_threshold)
        self.strategy_adapter = StrategyAdapter()
        self.meta = MetaCognitiveLayer(self.memory, self.strategy_adapter, reflection_frequency, failure_trigger)

    def handle_query(self, raw_query: str):
        processed = self.input_handler.preprocess(raw_query)
        context = {"memory_summary": self._summarize_memory_for_context()}
        proposals = self.cognitive.propose_actions(processed, context)
        attempted = 0
        for _ in range(3):
            if not proposals:
                break
            top = proposals[0]
            result = self.executor.execute(top)
            adjusted_conf, needs_replan = self.evaluator.evaluate(top, result)
            top_adj = copy.deepcopy(top)
            top_adj.confidence = adjusted_conf
            self.memory.store_action(raw_query, top_adj, result)
            attempted += 1
            adapted = self.meta.observe_and_maybe_reflect()
            if adapted:
                self.cognitive.update_strategies(adapted)
            if needs_replan and attempted < 3:
                proposals = self.cognitive.propose_actions(processed, context)
                continue
            else:
                break

    def _summarize_memory_for_context(self) -> Dict[str, Any]:
        recent = self.memory.recent(5)
        failures = sum(1 for e in recent if not e["result"].success)
        avg_conf = self.memory.average_confidence_recent(5) if recent else 1.0
        summary = {"recent_failures": failures, "avg_confidence_recent": avg_conf, "strategies": self.memory.get_strategies()}
        return summary

# -------------------- Demo --------------------

if __name__ == "__main__":
    agent = ModularAgent()
    queries = [
        "Fetch latest metrics for service X",
        "Optimize the allocation for task Y to reduce cost",
        "Explain why the deployment failed and how to fix it",
    ]
    for q in queries:
        print(f"\nQuery: {q}")
        agent.handle_query(q)
        print("Top memory entry:", agent.memory.actions_log[-1])
