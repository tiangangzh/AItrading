"""Learning from backtests: a ledger of hypotheses and evidence, Bayesian beliefs, judges, a research loop.

A strategy idea is a hypothesis about the market. Every backtest, holdout test and paper-trading
record is evidence about it. This package keeps all of it (:mod:`ledger`), turns it into beliefs
that related hypotheses share (:mod:`bayes`, :mod:`semantics`), lets a judge - the Bayesian engine,
a prompted LLM or a model you trained - assess a new idea against that experience before it is
tested (:mod:`judge`), scores every judge's pre-registered forecasts (:mod:`calibration`), exports
point-in-time training data for a judge model (:mod:`dataset`), and runs a propose -> judge -> test ->
learn loop with a final out-of-sample exam (:mod:`loop`). See docs/LEARNING_LOOP.md.
"""

from aitrading.beliefs.bayes import BeliefModel, Hyper
from aitrading.beliefs.digest import experience_digest, lessons
from aitrading.beliefs.judge import BayesianJudge, LLMJudge, OpenAICompatibleJudge
from aitrading.beliefs.learner import Learner
from aitrading.beliefs.ledger import BeliefLedger, default_beliefs_root
from aitrading.beliefs.loop import HeuristicGenerator, LLMGenerator, LoopConfig, ResearchLoop
from aitrading.beliefs.models import Assessment, Belief, BeliefUpdate, Evidence, Hypothesis, Prediction
from aitrading.beliefs.semantics import hypothesis_id, hypothesis_statement, similarity

__all__ = [
    "Assessment",
    "BayesianJudge",
    "Belief",
    "BeliefLedger",
    "BeliefModel",
    "BeliefUpdate",
    "Evidence",
    "HeuristicGenerator",
    "Hyper",
    "Hypothesis",
    "LLMGenerator",
    "LLMJudge",
    "Learner",
    "LoopConfig",
    "OpenAICompatibleJudge",
    "Prediction",
    "ResearchLoop",
    "default_beliefs_root",
    "experience_digest",
    "hypothesis_id",
    "hypothesis_statement",
    "lessons",
    "similarity",
]
