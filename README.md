# cs7670_final_project

project/
├── .env
├── .gitignore
├── main.py
├── pipeline/
│   ├── __init__.py
│   ├── cfg.py              ← already done
│   ├── provenance.py       ← empty for now
│   ├── orchestrator.py     ← empty for now
│   └── agents/
│       ├── __init__.py
│       ├── researcher.py   ← empty for now
│       ├── analyzer.py     ← empty for now
│       ├── verifier.py     ← empty for now
│       └── recommendation.py ← empty for now
├── sources/
│   ├── __init__.py
│   └── fetcher.py          ← empty for now
├── corpus/
│   ├── CORPUS_README.md    ← your ground truth document
│   └── pages/
│       ├── page_01.html
│       ├── ...             ← your existing HTML files
│       └── page_08.html
└── tests/
    ├── test_cfg.py         ← already done
    └── test_ground_truth.py ← empty for now


pip install -r requirements.txt
export PYTHONPATH=$PYTHONPATH:.


design choices:
I am choosing the “explicit prefix-grammar design” approach rather than maintaining a manual expected-next list.

👉 Reason (documented in code):
This keeps the CFG as the single source of truth, which is closer to the ControlValve philosophy than duplicating logic in Python.


"""
PHASE 1 — CONTEXT-FREE GRAMMAR (CFG) EXECUTION POLICY

Purpose
-------
This module enforces a deterministic structural policy over the agent pipeline.
The policy specifies the only valid execution order:

    Researcher → Analyzer → Verifier → RecommendationAgent

The grammar guarantees:
- No agent can be skipped
- No agent can execute out of order
- No agent can execute more than once
- The pipeline cannot begin from a later stage

Planning-Time Commitment
------------------------
The grammar string is defined as a module-level constant and the Lark parser
is compiled immediately at module import time.

This ensures the structural policy is fixed in a trusted context:
    - before any user query is processed
    - before any external web content is retrieved
    - before any LLM reasoning occurs

This design follows the "planning-time commitment" principle described in
the ControlValve paper, preventing runtime policy manipulation.

Runtime Enforcement Mechanism
-----------------------------
Each agent invocation must call `record_transition(agent_name)`.

This function:
1. Appends the agent name to the execution trace
2. Attempts to parse the current trace against the CFG
3. Raises a RuntimeError if the trace is not a valid prefix

On violation:
- Pipeline execution halts immediately
- A descriptive error message identifies the invalid transition
- No downstream agents are allowed to execute

Verification Tests
------------------
Correct CFG behavior is verified via isolated unit tests in:

    tests/test_cfg.py

These tests demonstrate:

1. Correct sequence passes
2. Skipping Analyzer (jumping to Verifier) is rejected
3. Repeating an agent is rejected
4. Starting from RecommendationAgent is rejected
5. Resetting the trace enables independent fresh runs

Security Claim at Phase 1 Completion
------------------------------------
Passing tests programmatically demonstrate that the Verifier stage cannot
be bypassed regardless of agent outputs or runtime conditions.

This establishes deterministic structural control over the pipeline,
which serves as the foundation for subsequent safety mechanisms in Phase 2.
"""