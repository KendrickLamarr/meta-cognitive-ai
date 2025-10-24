from __future__ import annotations
import logging
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import statistics
import copy

# Logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ModularAgent")


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


class InputHandler:
    """Receives raw user input and prepares structured data for the cognitive engine."""

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


class MemoryKB:
    """In-memory knowledge base."""

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


class CognitiveEngine:
    """Performs reasoning to propose actions."""

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

        external_scores = self._placeholder_external_model_score(processed_input, context)

        for i in range(self.params.get("max_proposals", 3)):
            name = f"{intent}_action_{i+1}"
            params = {"attempt": i + 1, "strategy": strat}
            ext = external_scores[i] if i < len(external_scores) else 0.0
            confidence = max(0.0, min(1.0, base_conf * (0.6 + 0.4 * ext) - i * 0.1))
            proposals.append(ProposedAction(name=name, params=params, confidence=confidence))
            logger.debug("[CognitiveEngine] Proposal %s: conf=%.3f", name, confidence)

        proposals.sort(key=lambda p: p.confidence, reverse=True)
        logger.info("[CognitiveEngine] Proposed %d actions: %s", len(proposals), proposals)
        return proposals

    def update_strategies(self, strategies: Dict[str, Any]):
        logger.info("[CognitiveEngine] Updating strategies: %s", strategies)
        self.strategies.update(strategies)

    def _select_strategy_for_intent(self, intent: str) -> Dict[str, Any]:
        strat = self.strategies.get(intent) or self.strategies.get("default", {"aggressiveness": 0.5})
        logger.debug("[CognitiveEngine] Selected strategy for intent '%s': %s", intent, strat)
        return strat

    def _placeholder_external_model_score(self, processed_input: Dict[str, Any], context: Dict[str, Any]) -> List[float]:
        length = processed_input.get("length", 0)
        seed = length % 97
        random.seed(seed)
        scores = [random.uniform(0.0, 1.0) for _ in range(self.params.get("max_proposals", 3))]
        logger.debug("[CognitiveEngine] External model placeholder scores: %s", scores)
        return scores


class ActionExecutor:
    """Executes a proposed action and returns an ActionResult."""

    def execute(self, action: ProposedAction) -> ActionResult:
        logger.info("[ActionExecutor] Executing action %s with params %s", action.name, action.params)
        success_chance = action.confidence
        success = random.random() < success_chance
        output = {"executed_action": action.name, "params": action.params, "simulated_success_probability": success_chance}
        result = ActionResult(success=success, output=output, details={"executed": True})
        logger.info("[ActionExecutor] Execution result: success=%s, details=%s", result.success, result.details)
        return result


class FeedbackEvaluator:
    """Evaluates action outcomes, adjusts confidence, and may request replanning."""

    def __init__(self, replanning_threshold: float = 0.6):
        self.replanning_threshold = replanning_threshold
        logger.debug("[FeedbackEvaluator] Initialized with replanning_threshold=%f", replanning_threshold)

    def evaluate(self, action: ProposedAction, result: ActionResult) -> Tuple[float, bool]:
        logger.info("[FeedbackEvaluator] Evaluating action %s (orig_conf=%.3f)", action.name, action.confidence)
        adjusted_confidence = action.confidence
        success = result.success

        if success:
            adjusted_confidence = min(1.0, action.confidence + 0.1)
        else:
            adjusted_confidence = max(0.0, action.confidence - 0.2)

        external_adjust = self._placeholder_external_evaluator(result)
        adjusted_confidence = max(0.0, min(1.0, adjusted_confidence * (1.0 + external_adjust)))

        needs_replan = adjusted_confidence < self.replanning_threshold
        logger.info("[FeedbackEvaluator] Adjusted_confidence=%.3f, needs_replan=%s", adjusted_confidence, needs_replan)
        return adjusted_confidence, needs_replan

    def _placeholder_external_evaluator(self, result: ActionResult) -> float:
        adj = (0.5 - random.random()) * 0.1
        logger.debug("[FeedbackEvaluator] External evaluator placeholder adj=%f", adj)
        return adj


class StrategyAdapter:
    """Adapts strategies based on self-reflection."""

    def adapt(self, memory: MemoryKB) -> Dict[str, Any]:
        logger.info("[StrategyAdapter] Running self-reflection to adapt strategies.")
        recent = memory.recent(20)
        adapted: Dict[str, Any] = {}

        failures = sum(1 for e in recent if not e["result"].success)
        avg_conf = memory.average_confidence_recent(20)

        logger.debug("[StrategyAdapter] recent_failures=%d, avg_conf=%.3f", failures, avg_conf)

        if failures >= 3 or avg_conf < 0.5:
            adapted["default"] = {"aggressiveness": max(0.1, 0.4)}
            logger.info("[StrategyAdapter] Adapting to more conservative strategy: %s", adapted["default"])
        else:
            adapted["default"] = {"aggressiveness": min(0.9, 0.6)}
            logger.info("[StrategyAdapter] Adapting to more aggressive strategy: %s", adapted["default"])

        memory.save_strategy("auto_adapted_default", adapted["default"])
        return adapted


class MetaCognitiveLayer:
    """Monitors system and triggers self-reflection and strategy adaptation."""

    def __init__(self, memory: MemoryKB, adapter: StrategyAdapter, reflection_frequency: int = 5, failure_trigger: int = 3):
        self.memory = memory
        self.adapter = adapter
        self.reflection_frequency = reflection_frequency
        self.failure_trigger = failure_trigger
        self.tasks_processed = 0
        logger.debug("[MetaCognitiveLayer] Initialized with freq=%d, failure_trigger=%d", reflection_frequency, failure_trigger)

    def observe_and_maybe_reflect(self) -> Optional[Dict[str, Any]]:
        self.tasks_processed += 1
        logger.debug("[MetaCognitiveLayer] Task processed count=%d", self.tasks_processed)

        if self.tasks_processed % self.reflection_frequency == 0:
            logger.info("[MetaCognitiveLayer] Periodic self-reflection triggered.")
            return self._reflect()

        recent_failures = self.memory.failures_in_recent(self.reflection_frequency)
        if recent_failures >= self.failure_trigger:
            logger.info("[MetaCognitiveLayer] Conditional self-reflection triggered due to failures=%d", recent_failures)
            return self._reflect()

        logger.debug("[MetaCognitiveLayer] No reflection triggered.")
        return None

    def _reflect(self) -> Dict[str, Any]:
        adapted = self.adapter.adapt(self.memory)
        logger.info("[MetaCognitiveLayer] Reflection produced adapted strategies: %s", adapted)
        return adapted


class ModularAgent:
    def __init__(self, replanning_threshold: float = 0.6, reflection_frequency: int = 5, failure_trigger: int = 3):
        self.input_handler = InputHandler()
        self.memory = MemoryKB()
        self.cognitive = CognitiveEngine()
        self.executor = ActionExecutor()
        self.evaluator = FeedbackEvaluator(replanning_threshold=replanning_threshold)
        self.strategy_adapter = StrategyAdapter()
        self.meta = MetaCognitiveLayer(memory=self.memory, adapter=self.strategy_adapter,
                                       reflection_frequency=reflection_frequency, failure_trigger=failure_trigger)
        logger.info("[ModularAgent] Agent initialized.")

    def handle_query(self, raw_query: str):
        logger.info("\n[ModularAgent] Handling new query: %s", raw_query)
        processed = self.input_handler.preprocess(raw_query)
        context = {"memory_summary": self._summarize_memory_for_context()}

        proposals = self.cognitive.propose_actions(processed, context)
        attempted = 0
        for attempt_limit in range(3):
            if not proposals:
                logger.warning("[ModularAgent] No proposals returned by CognitiveEngine.")
                break

            top = proposals[0]
            logger.info("[ModularAgent] Top proposal: %s (conf=%.3f)", top.name, top.confidence)

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
                logger.info("[ModularAgent] Confidence low (%.3f). Replanning...", adjusted_conf)
                self.cognitive.params["max_proposals"] = min(5, self.cognitive.params.get("max_proposals", 3) + 1)
                context["last_failure"] = {"attempt": attempted, "confidence": adjusted_conf, "result": result.output}
                proposals = self.cognitive.propose_actions(processed, context)
                continue
            else:
                logger.info("[ModularAgent] Accepting result. success=%s, adjusted_confidence=%.3f", result.success, adjusted_conf)
                break

    def _summarize_memory_for_context(self) -> Dict[str, Any]:
        recent = self.memory.recent(5)
        failures = sum(1 for e in recent if not e["result"].success)
        avg_conf = self.memory.average_confidence_recent(5) if recent else 1.0
        summary = {"recent_failures": failures, "avg_confidence_recent": avg_conf, "strategies": self.memory.get_strategies()}
        logger.debug("[ModularAgent] Memory summary for context: %s", summary)
        return summary


# ------------------------ Demo ------------------------
if __name__ == "__main__":
    logger.setLevel(logging.DEBUG)

    agent = ModularAgent(replanning_threshold=0.6, reflection_frequency=3, failure_trigger=2)

    queries = [
        "Fetch latest metrics for service X",
        "Optimize the allocation for task Y to reduce cost",
        "Explain why the deployment failed and how to fix it",
        "Optimize the allocation for task Y to reduce cost (again)",
        "Tune hyperparameters for model Z, it's complex and hard",
    ]

    for q in queries:
        agent.handle_query(q)

    logger.info("\n[Demo] Final memory entries:")
    for idx, entry in enumerate(agent.memory.actions_log, 1):
        logger.info("Entry %d: query='%s' action='%s' conf=%.3f success=%s",
                    idx, entry["query"], entry["action"].name, entry["confidence"], entry["result"].success)
