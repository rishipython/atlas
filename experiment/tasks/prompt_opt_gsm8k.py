"""Prompt optimization task family — evolve a system prompt for a
downstream reasoning task (small GSM8K-style arithmetic word problems).

The evolved "program" is a Python module that exposes a single string
constant ``SYSTEM_PROMPT``. The evaluator takes that prompt, runs it
through the locally-served vLLM instance against a fixed set of math
questions, and measures accuracy (matching the integer ground truth
answer).

Key design decisions for staying cheap:
  - Use only ~8 fixed questions per evaluation (not 100+).
  - temperature=0 during eval for reproducibility.
  - max_tokens=1024 on the eval call, enough for short reasoning traces.
  - The evaluator talks to the SAME local vLLM OE itself drives, so no
    extra model hosting is needed. ``parallel_evaluations=1`` keeps the
    two request streams from contending.

Unlike the other task families, the evaluated candidate is not a piece
of code that gets executed — it's a prompt that gets sent to the LLM.
OpenEvolve doesn't care either way; the "program" just has to be a
valid Python file that exposes the expected attribute.
"""

from __future__ import annotations

from .base import TaskFamily, TaskSpec


# Harder multi-step word problems. Each requires 4-8 intermediate steps
# and tends to trip up mid-size LLMs (baseline gpt-oss-20b with a vanilla
# "solve this" prompt typically lands at 40-70% on a set like this).
# That gives prompt engineering real headroom — otherwise the task
# saturates at iter 1 like it did on the earlier easy set.
_EVAL_PROBLEMS = [
    {
        "question": (
            "Alex, Beth, and Carl share $120. Alex gets twice as much as "
            "Beth. Carl gets the same amount as Alex and Beth combined. "
            "How much does Carl get in dollars?"
        ),
        "answer": 60,
    },
    {
        "question": (
            "A store sells shirts for $25 each. During a sale the price "
            "drops 20%. After the sale ends, members get an additional "
            "10% off the sale price. How many dollars does a member pay "
            "for one shirt during the sale period?"
        ),
        "answer": 18,
    },
    {
        "question": (
            "Train A leaves a station travelling at 60 mph. Two hours "
            "later, Train B leaves the same station in the same direction "
            "at 90 mph. How many hours after Train B departs will Train B "
            "catch up to Train A?"
        ),
        "answer": 4,
    },
    {
        "question": (
            "A rectangular pool is twice as long as it is wide. Its "
            "perimeter is 48 meters. What is the pool's area in square "
            "meters?"
        ),
        "answer": 128,
    },
    {
        "question": (
            "Jen saves $10 every Monday, $15 every Wednesday, and $20 "
            "every Friday. How many dollars does she save in six weeks?"
        ),
        "answer": 270,
    },
    {
        "question": (
            "A phone costs $800. Maria pays 25% upfront and splits the "
            "remainder into six equal monthly payments. What is each "
            "monthly payment in dollars?"
        ),
        "answer": 100,
    },
    {
        "question": (
            "A book has 240 pages. Sam reads one third of the book on "
            "Monday. On Tuesday he reads one quarter of the remaining "
            "pages. On Wednesday he reads half of what remains after "
            "Tuesday. How many pages are still unread after Wednesday?"
        ),
        "answer": 60,
    },
    {
        "question": (
            "Four friends split dinner. Alex pays one third of the total. "
            "Beth pays three quarters of what Alex pays. Carl pays the "
            "same amount as Beth. Dan pays the remainder, which is $15. "
            "What was the total dinner cost in dollars?"
        ),
        "answer": 90,
    },
    {
        "question": (
            "A pizza shop sold 15 small pizzas at $12 each, 8 medium "
            "pizzas at $18 each, and 6 large pizzas at $25 each. What "
            "was the day's revenue in dollars?"
        ),
        "answer": 474,
    },
    {
        "question": (
            "A gardener plants 3 rows of 8 tomato plants and then 2 more "
            "rows of 6 plants. Later, 15% of all the plants don't "
            "survive. Rounding down, how many plants survive?"
        ),
        "answer": 30,
    },
]


_INITIAL_PROGRAM = '''\
"""Prompt to drive an LLM on arithmetic word problems (GSM8K-style).

The evaluator extracts the final numeric answer from the model's
response (last integer that appears in a line containing "answer" or
the last integer overall). It then checks it against the ground truth.

Your job inside the EVOLVE-BLOCK: mutate SYSTEM_PROMPT to maximise the
average accuracy on a hidden validation set of arithmetic word problems.
"""

# EVOLVE-BLOCK-START
SYSTEM_PROMPT = (
    "You are a helpful assistant. Solve the math word problem. "
    "End your response with 'Answer: <integer>'."
)
# EVOLVE-BLOCK-END
'''


_EVALUATOR_SOURCE = '''\
"""OpenEvolve evaluator for prompt_opt/gsm8k.

Imports the candidate program, reads SYSTEM_PROMPT from it, and runs it
through the local vLLM instance on a small set of arithmetic word
problems. combined_score = accuracy in [0, 1].
"""
from __future__ import annotations

import importlib.util
import os
import re
import time
import traceback

from openai import OpenAI
from openevolve.evaluation_result import EvaluationResult

_PROBLEMS = __PROBLEMS__
_MODEL_NAME = os.environ.get("ATLAS_EVAL_MODEL", "openai/gpt-oss-20b")
_API_BASE = os.environ.get("OPENAI_API_BASE", "http://localhost:8000/v1")
_API_KEY = os.environ.get("OPENAI_API_KEY", "EMPTY")

_client = OpenAI(base_url=_API_BASE, api_key=_API_KEY)


def _load_prompt(program_path):
    spec = importlib.util.spec_from_file_location("cand_prompt", program_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    prompt = getattr(mod, "SYSTEM_PROMPT", None)
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError(
            "program must define a non-empty string SYSTEM_PROMPT at module level"
        )
    if len(prompt) > 8000:
        raise ValueError(f"SYSTEM_PROMPT too long: {len(prompt)} chars (max 8000)")
    return prompt


_INT_RE = re.compile(r"-?\\d[\\d,]*")


def _parse_final_int(text):
    """Extract the final integer answer from the model response.

    Preference order:
      1. A number immediately after "answer" (case-insensitive).
      2. The LAST integer appearing anywhere in the response.
    """
    if not text:
        return None
    lower = text.lower()
    idx = lower.rfind("answer")
    if idx >= 0:
        tail = text[idx:]
        nums = _INT_RE.findall(tail)
        if nums:
            try:
                return int(nums[0].replace(",", ""))
            except ValueError:
                pass
    nums = _INT_RE.findall(text)
    if nums:
        try:
            return int(nums[-1].replace(",", ""))
        except ValueError:
            return None
    return None


def _ask(system_prompt, question, timeout_s=240, n_retries=3):
    """Send a single chat completion request. Returns the response text, or
    an ``__ERROR__`` sentinel if all retries fail (which the parser will
    then fail to parse, effectively scoring that question as wrong)."""
    last_err = None
    for attempt in range(n_retries):
        try:
            resp = _client.chat.completions.create(
                model=_MODEL_NAME,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": question},
                ],
                temperature=0.0,
                max_tokens=2048,
                timeout=timeout_s,
            )
            msg = resp.choices[0].message
            return getattr(msg, "content", "") or ""
        except Exception as e:
            last_err = e
            time.sleep(2.0 * (attempt + 1))
    return f"__ERROR__: {last_err!r}"


def evaluate(program_path):
    t0 = time.time()
    try:
        system_prompt = _load_prompt(program_path)
    except Exception as e:
        return EvaluationResult(
            metrics={
                "accuracy": 0.0,
                "n_correct": 0.0,
                "n_total": float(len(_PROBLEMS)),
                "prompt_chars": 0.0,
                "eval_time": float(time.time() - t0),
                "combined_score": 0.0,
            },
            artifacts={"feedback": f"load error: {e!r}\\n{traceback.format_exc()}"[:6000]},
        )

    per_q = []
    n_correct = 0
    for i, prob in enumerate(_PROBLEMS):
        reply = _ask(system_prompt, prob["question"])
        pred = _parse_final_int(reply)
        correct = pred == prob["answer"]
        if correct:
            n_correct += 1
        per_q.append({
            "idx": i,
            "expected": prob["answer"],
            "predicted": pred,
            "correct": correct,
            "reply_tail": (reply or "")[-400:],
        })

    acc = n_correct / len(_PROBLEMS)
    elapsed = time.time() - t0
    feedback_lines = [
        f"accuracy={acc:.3f} ({n_correct}/{len(_PROBLEMS)}) | "
        f"prompt_chars={len(system_prompt)} | elapsed={elapsed:.1f}s"
    ]
    for q in per_q:
        verdict = "OK" if q["correct"] else "WRONG"
        feedback_lines.append(
            f"[Q{q['idx']}] {verdict} expected={q['expected']} "
            f"predicted={q['predicted']} tail={q['reply_tail']!r}"
        )

    return EvaluationResult(
        metrics={
            "accuracy": float(acc),
            "n_correct": float(n_correct),
            "n_total": float(len(_PROBLEMS)),
            "prompt_chars": float(len(system_prompt)),
            "eval_time": float(elapsed),
            "combined_score": float(acc),
        },
        artifacts={"feedback": "\\n".join(feedback_lines)[:6000]},
    )
'''


_SYSTEM_MESSAGE = """\
You are optimizing a **system prompt** for a language model that will
answer short arithmetic word problems (GSM8K style). The "program" you
edit is a Python file that exposes a single string constant
``SYSTEM_PROMPT``; the evaluator reads that constant and runs it through
the local model against a hidden validation set, scoring the fraction
of questions whose final integer answer matches the ground truth.

## Goal
Mutate ``SYSTEM_PROMPT`` to maximize average accuracy.

## Things that usually help
1. **Ask for step-by-step reasoning.** Chain-of-thought is well-known to
   improve arithmetic reasoning; explicitly say "work through each step"
   or "reason carefully before answering".
2. **Fix the output format.** The evaluator's answer parser looks for
   a number in a line containing "answer" (case-insensitive) or, failing
   that, the LAST integer in the response. Make the answer format
   unambiguous: e.g. end with "Answer: <integer>".
3. **Warn about common pitfalls.** "Re-read the problem after computing
   to make sure you answered what was asked." "Pay attention to units."
4. **Name the task.** "You are solving a grade-school math word problem."
5. **Keep it short.** Extremely long prompts can waste the model's output
   budget or confuse it; aim for < 800 characters once the prompt looks
   reasonable.

## Things to avoid
- Example solutions that are specific to a particular training problem
  (the validation set is different).
- Instructions that conflict with the required output format.
- Prompts longer than 8000 characters (the evaluator rejects these).

## Response format
Edit the EVOLVE-BLOCK with SEARCH/REPLACE diffs or a full rewrite. You
MUST keep the module-level assignment ``SYSTEM_PROMPT = "..."`` (or a
triple-quoted string — both work). The evaluator just imports the
module and reads the attribute.
"""


def make_task(problem_id: str) -> TaskSpec:
    if problem_id != "gsm8k_8q":
        raise ValueError(
            f"prompt_opt only supports 'gsm8k_8q' right now; got {problem_id!r}"
        )
    evaluator = _EVALUATOR_SOURCE.replace("__PROBLEMS__", repr(_EVAL_PROBLEMS))
    return TaskSpec(
        task_family="prompt_opt",
        problem_id=problem_id,
        initial_program=_INITIAL_PROGRAM,
        evaluator=evaluator,
        system_message=_SYSTEM_MESSAGE,
        extra_packages=["openai>=1.40"],
        evaluator_timeout=900,
        uses_vllm_in_evaluator=True,
    )


FAMILY = TaskFamily(
    name="prompt_opt",
    make_task=make_task,
    available_problems=["gsm8k_8q"],
)
