# Redesigned Architecture: AgentCore + Claude Code + Harness Kit

## Three-Layer Architecture for Matt

```
┌──────────────────────────────────────────────────────┐
│  HARNESS ENGINEERING KIT (Layer 3: Quality)          │
│                                                      │
│  PretestValidator: import check, PHI scan, deps      │
│  Nightly Tests: syntax, pytest, Dockerfile           │
│  Feedback Injection: errors → next day planning      │
│  Phase Gates: Foundation → Training → Integration    │
├──────────────────────────────────────────────────────┤
│  AGENTCORE MANAGED HARNESS (Layer 2: Governance)     │
│                                                      │
│  Orchestration: Planner assigns → Workers execute    │
│  Cedar Policies: file-scope writes, PHI protection   │
│  Memory: cross-session context persistence           │
│  Identity: agent auth + audit trail                  │
│  Firecracker: microVM hardware isolation per agent   │
├──────────────────────────────────────────────────────┤
│  CLAUDE CODE WORKERS (Layer 1: Execution)            │
│                                                      │
│  Each worker agent = Claude Code session             │
│  Write → Run → Fix → Rerun (self-debugging loop)    │
│  Shell access (sandboxed), file editing, testing     │
│  Real code, real tests, real iteration               │
├──────────────────────────────────────────────────────┤
│  BEDROCK MODELS                                      │
│  Opus 4.6 (Planner, Judge) / Sonnet 4.6 (Workers)   │
└──────────────────────────────────────────────────────┘
```

## How It Works — Healthcare Sprint

### Step 1: AgentCore orchestrates
```
AgentCore Planner (Opus):
  "Day 1: Assign US-101 (Patient Risk API) to clinical_ml_worker"
  "Day 1: Assign US-102 (Clinical Dashboard) to frontend_worker"
```

### Step 2: Cedar checks permissions BEFORE execution
```
Cedar Policy Engine:
  clinical_ml_worker → write src/risk_scoring/** → ALLOW ✓
  clinical_ml_worker → write src/compliance/** → DENY ✗
  frontend_worker → write src/frontend/** → ALLOW ✓
```

### Step 3: Claude Code executes the work
```
Claude Code Session (clinical_ml_worker):
  > "Build patient risk scoring API in src/risk_scoring/"
  
  Writing src/risk_scoring/model.py...
  Writing src/risk_scoring/api.py...
  Running: python -m pytest tests/test_risk_scoring.py
  FAIL: ImportError: sklearn not found
  Fixing: adding scikit-learn to requirements.txt
  Running: python -m pytest tests/test_risk_scoring.py
  PASS: 12/12 tests passing
  
  [Claude Code self-debugs — this is the real magic]
```

### Step 4: Harness validates the output
```
PretestValidator:
  ✓ Import check: all imports resolve
  ✓ PHI scan: no unencrypted patient data
  ✗ Dependency: sklearn not in pyproject.toml (only in requirements.txt)
  → Feedback: "Add scikit-learn to pyproject.toml dependencies"
```

### Step 5: Nightly tests + feedback loop
```
Nightly: 247 tests, 217 passed (87.9%)
Feedback → Day 2 planning → Claude Code fixes → 238 passed (96.4%)
```

## Why This Is Better Than Pure Strands Tools

| | Strands write_file tool | Claude Code worker |
|---|---|---|
| Code generation | Model generates string → tool writes | Claude Code writes, runs, debugs iteratively |
| Self-repair | None — needs nightly feedback | Built-in: write → test → fix → retest |
| Shell access | Limited sandbox | Full sandboxed shell |
| Realism | "Agent wrote a file" | "Agent built and tested a feature" |
| Demo impact | "It generated code" | "It built working software" |
| Matt's reaction | "Nice" | "This is production-ready" |

## Key Message Update

**Before:** "AgentCore runs the agents. Harness improves reasoning."
**Now:** "AgentCore governs WHAT gets built. Claude Code builds it. The harness ensures it's built RIGHT."

Three layers, three jobs:
1. **AgentCore** = the project manager (assigns work, enforces policies)  
2. **Claude Code** = the developer (writes code, runs tests, self-debugs)
3. **Harness Kit** = the QA team (validates quality, catches errors, provides feedback)

## Demo Changes

### Healthcare Demo — Updated Flow
- Planner on AgentCore assigns stories
- Cedar checks write permissions
- **Each worker spawns a Claude Code session** (ACP)
- Claude Code writes + tests + self-debugs
- Harness validates output (PretestValidator, PHI scan)
- Nightly tests → feedback → Day 2

### Trainium Demo — Already Fits
- The kernel-forge agent already uses the compile→verify→profile→reason loop
- This IS the Claude Code pattern: iterate on code using tool feedback
- No change needed for Trainium narrative

## What This Means for AgentCorp V9
- V8: Strands agents with write_file tools
- **V9: AgentCore orchestrator + Claude Code workers + Harness Kit**
- Each domain pack specifies worker type: `worker_engine: claude_code`
- AgentCore Managed Harness handles the orchestration
- Claude Code ACP sessions handle the execution
- Harness Kit validates between iterations
