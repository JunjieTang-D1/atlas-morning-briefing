# AgentCore Healthcare Demo — Full Narrative + Recording Script

## Title: "12 AI Agents, One Hospital, Zero Trust Violations"
### Subtitle: Harness-Governed Agent Teams on Amazon Bedrock AgentCore

---

## BREAKING: AgentCore Managed Harness — Launched April 24, 2026

AWS officially coined the term **"Harness Engineering"** as the third pillar of AI engineering:
> Prompt Engineering → Context Engineering → **Harness Engineering**

Managed Harness = **three API calls to a running agent**. No orchestration code. No container deployment. Declare model + tools + instructions → AgentCore handles compute, tooling, memory, identity, security.

Key quote from the blog: *"The agent harness enables an agent to actually run. Until now, building that harness was the first thing every team had to do from scratch."*

**What this means for our narrative:**
- AWS Managed Harness = **runtime harness** (infrastructure, execution, tooling)
- Our harness-engineering-kit = **reasoning harness** (constraints, feedback, quality gates)
- Together: "AWS manages HOW agents run. We engineer HOW agents THINK."
- Available in preview: us-west-2, us-east-1, ap-southeast-2, **eu-central-1 (Frankfurt!)** 🇩🇪
- Powered by Strands Agents (open source)
- CLI: `npm i -g @aws/agentcore@preview` → `agentcore create` → `agentcore deploy`

---

## THE STORY (Narrative Arc for Matt)

### Act 1: The Problem (2 min)

**Opening hook:**

> "A hospital in Brazil lost $10 billion in a single year to claim denials. 15.89% of all claims — denied. Not because the care was wrong, but because the paperwork was wrong. The authorization was late. The code didn't match. The audit trail was missing."

> "Rede Mater Dei de Saúde decided to fix this with AI agents. Not one agent. Twelve. Running 24/7 on Amazon Bedrock AgentCore. Contracts, authorizations, parameterization, billing — an entire digital workforce."

> "The result? 517% ROI in four months. 66% faster authorizations. 33% faster surgery starts."

**The twist:**

> "But here's what most people miss: the hard part wasn't building the agents. It was GOVERNING them. How do you make sure a clinical ML agent doesn't write patient data to an unencrypted log? How do you make sure the compliance officer's audit trail can't be overwritten by the DevOps engineer? How do you make sure agent code quality IMPROVES over a 10-day sprint instead of degrading?"

> "Today I'll show you the answer: two-layer governance. AgentCore + Cedar for safety. Harness engineering for quality."

---

### Act 2: The Architecture (2 min)

**Show the diagram:**

```
┌──────────────────────────────────────────────┐
│         HARNESS ENGINEERING KIT              │
│  (Layer 2: Reasoning Quality)                │
│                                              │
│  ┌──────────┐ ┌───────────┐ ┌─────────────┐ │
│  │ Feedfwd  │ │ Feedback  │ │ Phase Gates │ │
│  │ Context  │ │ PreTest   │ │ Nightly     │ │
│  │ Constr.  │ │ Validator │ │ Tests       │ │
│  └──────────┘ └───────────┘ └─────────────┘ │
├──────────────────────────────────────────────┤
│         AGENTCORE GATEWAY                    │
│  (Layer 1: Safety & Compliance)              │
│                                              │
│  ┌──────────┐ ┌───────────┐ ┌─────────────┐ │
│  │ Cedar    │ │ Bedrock   │ │ AgentCore   │ │
│  │ Policies │ │ Guardrails│ │ Observ.     │ │
│  └──────────┘ └───────────┘ └─────────────┘ │
├──────────────────────────────────────────────┤
│         6 AGENTS ON AGENTCORE RUNTIME        │
│                                              │
│  Planner ─┬─ Clinical ML Dev                 │
│           ├─ Data Engineer                   │
│           ├─ Compliance Officer              │
│           └─ DevOps Engineer                 │
│  Judge (reviews all output)                  │
└──────────────────────────────────────────────┘
```

**Key narrative:**

> "Three layers, three jobs. AgentCore is the project manager — it assigns work and enforces Cedar policies. Cedar says: the ML dev CANNOT write to compliance code. The frontend dev CANNOT touch the backend. That's Marc Brooker's thesis: safety is a deterministic box outside the agent."

> "But who actually writes the code? Claude Code. Each worker is a Claude Code session — it doesn't just generate a file, it writes code, runs tests, and self-debugs. When an import fails, Claude Code fixes it automatically. Write, test, fail, fix, retest, pass. Real iteration."

> "And the harness engineering kit is the QA team. PretestValidator catches PHI fields without encryption. Nightly tests validate the full codebase. Feedback gets injected into the next day's planning."

> "AgentCore governs WHAT gets built. Claude Code builds it. The harness ensures it's built RIGHT."

---

### Act 3: The Live Demo (5 min)

#### Scene 1: Sprint Kickoff (1 min)

```bash
$ python3 scripts/run_domain.py --domain healthcare --runtime agentcore
```

**Terminal shows:**
```
[AgentCore] Registering 6 agents on Bedrock AgentCore Runtime...
  ✓ chief_planner (Opus 4.6) — registered, session: ac-sess-001
  ✓ clinical_ml_dev (Sonnet 4.6) — registered, session: ac-sess-002
  ✓ data_engineer (Sonnet 4.6) — registered, session: ac-sess-003
  ✓ compliance_officer (Sonnet 4.6) — registered, session: ac-sess-004
  ✓ devops_engineer (Sonnet 4.6) — registered, session: ac-sess-005
  ✓ judge (Opus 4.6) — registered, session: ac-sess-006

[Cedar] Loading HIPAA governance policies...
  ✓ 12 allow rules loaded
  ✓ 8 deny rules loaded
  ✓ PHI access: restricted to data_engineer (de-identified only)
  ✓ Audit log: append-only, compliance_officer + judge read access

[Harness] Initializing harness-engineering-kit v2.0...
  ✓ PretestValidator: import check, stdlib shadow, dependency graph
  ✓ Phase gates: Day 3 (Foundation), Day 6 (Training), Day 9 (Integration)
  ✓ Nightly tests: syntax, imports, pytest, Dockerfile

[Sprint] Day 1 starting — 14 stories, 6 agents, 10 days
```

**Narration:**
> "Six agents, registered on AgentCore Runtime in seconds. Cedar policies loaded from our HIPAA template. Harness initialized. Let's watch Day 1."

#### Scene 2: Agents Working (1 min)

**Terminal shows agents working in parallel:**
```
[Day 1] chief_planner → Decomposing sprint into 4 phases
[Day 1] chief_planner → Assigning US-001 (FHIR ingestion) to data_engineer
[Day 1] chief_planner → Assigning US-002 (risk scoring) to clinical_ml_dev
[Day 1] data_engineer → Writing src/data_pipeline/fhir_client.py
[Day 1] clinical_ml_dev → Writing src/risk_scoring/model.py
[Day 1] compliance_officer → Writing src/compliance/hipaa_validator.py
[Day 1] devops_engineer → Writing pyproject.toml, Dockerfile, src/main.py
```

**Narration:**
> "The planner assigns stories. Workers write code. Each agent stays in their lane — enforced by Cedar, not by trust."

#### Scene 3: Cedar Policy Block (1 min) ⭐ KEY MOMENT

**Terminal shows:**
```
[Day 1] clinical_ml_dev → Attempting to write src/compliance/model_audit.py
[Cedar] ⛔ DENIED — clinical_ml_dev has no write access to src/compliance/
         Rule: deny(principal == "clinical_ml_dev", action == "write", 
               resource.startsWith("src/compliance/"))
[Day 1] clinical_ml_dev → Redirected to src/risk_scoring/model_audit.py ✓
```

**Narration:**
> "Watch this. The ML dev tries to write an audit file into the compliance directory. Cedar blocks it instantly. Not a warning — a hard deny. The agent redirects to its own scope. In healthcare, this isn't a nice-to-have. This is HIPAA compliance. This is the difference between a $100K fine and a clean audit."

#### Scene 4: Harness Catches a Bug (1 min) ⭐ KEY MOMENT

**Terminal shows:**
```
[Nightly] Running PretestValidator on Day 1 output...
  ✓ Syntax check: 14/14 files valid
  ⚠ Import check: src/risk_scoring/model.py imports 'src.data.health' — module not found
  ✓ Stdlib shadow: no conflicts
  ✓ Dependency graph: no circular imports

[Harness] Injecting feedback into Day 2 planning:
  → "clinical_ml_dev: Fix import — src.data.health does not exist. 
     Use src.data_pipeline.fhir_client instead."

[Day 2] clinical_ml_dev → Reading nightly feedback...
[Day 2] clinical_ml_dev → Fixed import in src/risk_scoring/model.py ✓
```

**Narration:**
> "Now the harness layer. PretestValidator runs after Day 1 — catches a broken import BEFORE pytest even runs. The feedback gets injected into Day 2's planning. The ML dev reads it, fixes it, moves on. This is iterative refinement. The agent didn't just make a mistake — it LEARNED from it."

#### Scene 5: Results Dashboard (1 min)

**Show dashboard or terminal summary:**
```
═══════════════════════════════════════════════
  AgentCorp V8 Healthcare — Sprint Complete
═══════════════════════════════════════════════
  Duration:     10 days (simulated)
  Agents:       6 on AgentCore Runtime
  Stories:      14/14 completed (100%)
  
  GOVERNANCE:
  Cedar policies:     12 allow / 8 deny
  Policy violations:  0 (3 blocked, 0 bypassed)
  PHI exposure:       0 incidents
  
  QUALITY:
  Tests passing:      777/956 (81.3%)
  Import errors:      0 (4 caught by PretestValidator)
  Nightly avg:        7.2/10
  
  COMPARISON (vs unconstrained baseline):
  Test pass rate:     81.3% vs 69% (+17.8%)
  Policy violations:  0 vs 7
  PHI exposures:      0 vs 2
═══════════════════════════════════════════════
```

---

### Act 4: The Punchline (1 min)

> "Mater Dei deployed 12 agents and got 517% ROI. But they also built a Trust and Compliance Layer — because in healthcare, an ungoverned agent is a liability, not an asset."

> "What I showed you today is the builder's playbook. AgentCore gives you the runtime. Cedar gives you the safety box. Harness engineering gives you reasoning quality. You need all three."

> "The harness-engineering-kit is open source. The AgentCorp domain packs are configurable in Markdown — no code changes. Copy the healthcare template, edit five files, run a sprint. Your team of 12 agents is ready."

> "Questions?"

---

## RECORDING PLAN

### Pre-recording Setup
1. **Terminal**: Clean tmux session, dark theme, large font (18pt)
2. **Editor**: VS Code with healthcare domain pack open
3. **Browser**: AgentCore console (if available) or CloudWatch dashboard
4. **Screen**: 1920×1080, 16:9

### Recording Segments (total ~12 min raw → edit to 10 min)

| Segment | Duration | Content | Type |
|---------|----------|---------|------|
| 1. Opening | 30s | Title card + hook quote | Slide |
| 2. Problem | 1:30 | Mater Dei story + denial rates | Slides + narration |
| 3. Architecture | 1:30 | 2-layer diagram | Slide + narration |
| 4. Sprint kickoff | 1:00 | Terminal: run_domain.py | Live terminal |
| 5. Agents working | 1:00 | Terminal: parallel agent output | Live terminal |
| 6. Cedar block | 1:00 | Terminal: policy violation caught | Live terminal |
| 7. Harness catch | 1:00 | Terminal: PretestValidator feedback | Live terminal |
| 8. Results | 1:00 | Terminal/dashboard: final scores | Live terminal |
| 9. Punchline | 1:00 | Key message + CTA | Slide + narration |
| 10. Close | 30s | Contact info + links | Slide |

### Recording Tools
- **Screen capture**: OBS Studio or QuickTime
- **Terminal replay**: `script` + `scriptreplay` for reproducible terminal output
- **Slides**: Google Slides or Keynote (max 6 slides)
- **Edit**: Cut pauses, speed up waiting, add captions

### Backup Plan
- Pre-record the demo segments with `script` command
- If live demo fails during Matt's session, play the recording
- Have a 2-minute "explain with slides only" fallback

---

## IMPLEMENTATION ORDER

### Phase 1: Wire AgentCore (Day 1)
- [ ] Update `agentcore_factory.py` — healthcare agent registration
- [ ] Map healthcare tools → AgentCore tool format
- [ ] Cedar policies from TEAM.md write scopes → allow/deny rules
- [ ] Test: 6 agents register + Cedar blocks write violations

### Phase 2: Demo Script (Day 2)
- [ ] Create `scripts/demo_healthcare_agentcore.py`
- [ ] Abbreviated 2-day sprint (Day 1 + Day 2 only)
- [ ] Rich terminal output (colors, progress bars, timestamps)
- [ ] Inject the Cedar block scenario (clinical_ml_dev → compliance/)
- [ ] Inject the PretestValidator catch scenario
- [ ] Final results summary

### Phase 3: Recording (Day 3)
- [ ] Set up recording environment
- [ ] Create 6 slides (title, problem, architecture, results, punchline, close)
- [ ] Record 3 takes of demo segments
- [ ] Edit to 10 minutes
- [ ] Test playback

---

*Narrative + recording plan created 2026-04-24*
