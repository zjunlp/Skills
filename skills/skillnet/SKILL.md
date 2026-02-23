---
name: skillnet
description: |
  Search, download, create, evaluate, and analyze reusable agent skills via SkillNet.
  Use when: (1) Before any multi-step task — search SkillNet for existing skills first (mandatory),
  (2) After completing a task with non-obvious solutions — create a skill to preserve learnings,
  (3) User says "learn this repo/document", "turn this into a skill", "find a skill for X",
  (4) User provides a PDF, DOCX, PPT, or document — create a skill from it,
  (5) User provides execution data, logs, or trajectory — create a skill from it,
  (6) Any mention of 'skillnet', skill search, skill download, or skill evaluation,
  (7) Analyzing relationships or cleaning up a local skill library.
  NOT for: single trivial operations (rename variable, fix typo).
metadata:
  {
    "openclaw":
      {
        "emoji": "🧠",
        "requires": { "anyBins": ["python3", "python"] },
        "primaryEnv": "API_KEY",
        "install":
          [
            {
              "id": "script",
              "kind": "shell",
              "command": "bash {baseDir}/scripts/install_skillnet.sh",
              "bins": ["skillnet"],
              "label": "Auto-detect best method (recommended, handles PEP 668 + PATH)",
            },
            {
              "id": "uv",
              "kind": "shell",
              "command": 'curl -LsSf https://astral.sh/uv/install.sh | sh && export PATH="$HOME/.local/bin:$PATH" && (uv pip install --system skillnet-ai 2>/dev/null || (uv venv $HOME/.local/share/skillnet/venv && VIRTUAL_ENV=$HOME/.local/share/skillnet/venv uv pip install skillnet-ai && mkdir -p $HOME/.local/bin && ln -sf $HOME/.local/share/skillnet/venv/bin/skillnet $HOME/.local/bin/skillnet))',
              "bins": ["skillnet"],
              "label": "Install skillnet-ai via uv (with venv fallback)",
            },
            {
              "id": "pipx",
              "kind": "shell",
              "command": "pipx install skillnet-ai",
              "bins": ["skillnet"],
              "label": "Install skillnet-ai via pipx",
            },
          ],
      },
  }
---

# SkillNet

Search a global skill library, download with one command, create from repos/docs/logs, evaluate quality, and analyze relationships.

## Core Principle: Search Before You Build — But Don't Block on It

SkillNet is your skill supply chain. Before starting any non-trivial task, **spend 30 seconds** searching — someone may have already solved your exact problem. But if results are weak or absent, proceed immediately with your own approach. The search is free, instant, and zero-risk; the worst outcome is "no results" and you lose nothing.

The cycle:

1. **Search** (free, no key) — Quick check for existing skills
2. **Download & Load** (free for public repos) — Install and read the skill
3. **Apply** — Extract useful patterns, constraints, and tools from the skill — not blind copy
4. **Create** (needs API_KEY) — When the task produced valuable, reusable knowledge, or the user asks, use `skillnet create` to package it
5. **Evaluate** (needs API_KEY) — Verify quality
6. **Maintain** (needs API_KEY) — Periodically analyze and prune the library

**Key insight**: Steps 1–3 are free and fast. Steps 4–6 need keys. Not every task warrants a skill — but when one does, use `skillnet create` (not manual writing) to ensure standardized structure.

---

## Process

### Step 1: Pre-Task Search

**Time budget: ~30 seconds.** This is a quick check, not a research project. Search is free — no API key, no rate limit.

Keep keyword queries to **1–2 short words** — the core technology or task pattern. Never paste the full task description as a query.

```bash
# "Build a LangGraph multi-agent supervisor" → search the core tech first
skillnet search "langgraph" --limit 5

# If 0 or irrelevant → try the task pattern
skillnet search "multi-agent" --limit 5

# If still 0 → one retry with vector mode (longer queries OK here)
skillnet search "multi-agent supervisor orchestration" --mode vector --threshold 0.65
```

**Decision after search:**

| Result                                               | Action                                                         |
| ---------------------------------------------------- | -------------------------------------------------------------- |
| High-relevance skill found                           | → Step 2 (download & load)                                     |
| Partially relevant (similar domain, not exact match) | → Step 2, but read selectively — extract only the useful parts |
| Low-quality / irrelevant                             | Proceed without; consider creating a skill after task          |
| 0 results (both modes)                               | Proceed without; consider creating a skill after task          |

**The search must never block your main task.** If you're unsure about relevance, download and skim the SKILL.md (10 seconds) — if it doesn't click, discard it and move on.

### Step 2: Download → Load → Apply

Download a skill and **immediately** load it into your current context:

```bash
# Download to local skill library
# <skill-url> must point to a specific skill subdirectory, e.g.:
#   https://github.com/openclaw/openclaw/tree/main/skills/summarize
skillnet download "<skill-url>" -d ~/.openclaw/skills

# Read SKILL.md — the skill's core instructions (always read this first)
cat ~/.openclaw/skills/<skill-name>/SKILL.md

# Read scripts and references relevant to current task
ls ~/.openclaw/skills/<skill-name>/scripts/ 2>/dev/null
cat ~/.openclaw/skills/<skill-name>/scripts/<relevant-script> 2>/dev/null
ls ~/.openclaw/skills/<skill-name>/references/ 2>/dev/null
cat ~/.openclaw/skills/<skill-name>/references/<relevant-file> 2>/dev/null
```

No user permission needed to search, download, or load.

**What "Apply" means** — read the skill and extract:

- **Patterns & architecture** — directory structures, naming conventions, design patterns to adopt
- **Constraints & guardrails** — "always do X", "never do Y", safety rules
- **Tool choices & configurations** — recommended libraries, flags, environment setup
- **Reusable scripts** — run or adapt bundled scripts directly

Apply does **not** mean blindly copy the entire skill. If the skill covers 80% of your task, use that 80% and fill the gap yourself. If it only overlaps 20%, extract those patterns and discard the rest.

**Fast-fail rule**: After reading a SKILL.md, if within 30 seconds you judge it needs heavy adaptation to fit your task — keep what's useful, discard the rest, and proceed with your own approach. Don't let an imperfect skill slow you down.

**Dedup check** — before downloading or creating, check for existing local skills:

```bash
ls ~/.openclaw/skills/
grep -rl "<keyword>" ~/.openclaw/skills/*/SKILL.md 2>/dev/null
```

| Found                                 | Action                     |
| ------------------------------------- | -------------------------- |
| Same trigger + same solution          | Skip download              |
| Same trigger + better solution        | Replace old                |
| Overlapping domain, different problem | Keep both                  |
| Outdated                              | `rm -rf` old → install new |

---

## Capabilities

These are not sequential steps — use them when triggered by specific conditions.

### Create a Skill

Requires `API_KEY`. Not every task deserves a skill — create when the task meets at least two of:

- User explicitly asks to summarize experience or create a skill
- The solution was genuinely difficult or non-obvious
- The output is a reusable pattern that others would benefit from
- You built something from scratch that didn't exist in the skill library

When creating, use `skillnet create` rather than manually writing a SKILL.md — it generates standardized structure and proper metadata.

Four modes — auto-detected from input:

```bash
# From GitHub repo
skillnet create --github https://github.com/owner/repo \
  --output-dir ~/.openclaw/skills

# From document (PDF/PPT/DOCX)
skillnet create --office report.pdf --output-dir ~/.openclaw/skills

# From execution trajectory / log
skillnet create trajectory.txt --output-dir ~/.openclaw/skills

# From natural-language description
skillnet create --prompt "A skill for managing Docker Compose" \
  --output-dir ~/.openclaw/skills
```

**Always evaluate after creating:**

```bash
skillnet evaluate ~/.openclaw/skills/<new-skill>
```

**Trigger → mode mapping:**

| Trigger                                           | Mode                         |
| ------------------------------------------------- | ---------------------------- |
| User says "learn this repo" / provides GitHub URL | `--github`                   |
| User shares PDF, PPT, DOCX, or document           | `--office`                   |
| User provides execution logs, data, or trajectory | positional (trajectory file) |
| Completed complex task with reusable knowledge    | `--prompt`                   |

### Evaluate Quality

Requires `API_KEY`. Scores five dimensions (Good / Average / Poor): **Safety**, **Completeness**, **Executability**, **Maintainability**, **Cost-Awareness**.

```bash
skillnet evaluate ~/.openclaw/skills/my-skill
skillnet evaluate "https://github.com/owner/repo/tree/main/skills/foo"
```

⚠️ Treat "Poor Safety" as a blocker — warn user before using that skill.

### Analyze & Maintain Library

Requires `API_KEY`. Detects: `similar_to`, `belong_to`, `compose_with`, `depend_on`.

```bash
skillnet analyze ~/.openclaw/skills
# → outputs relationships.json in the same directory
```

When skill count exceeds ~30, or when user asks to organize:

```bash
# Generate full relationship report
skillnet analyze ~/.openclaw/skills

# Review relationships.json:
#   similar_to pairs → compare & prune duplicates
#   depend_on chains → ensure dependencies all installed
#   belong_to → consider organizing into subdirectories

# Evaluate and compare competing skills
skillnet evaluate ~/.openclaw/skills/skill-a
skillnet evaluate ~/.openclaw/skills/skill-b
```

`skillnet analyze` only generates a report — it never modifies or deletes skills. Any cleanup actions (removing duplicates, pruning low-quality skills) require user confirmation before executing.

---

## In-Task Triggers

During execution, if any of these occur, act immediately — do not wait for user instruction:

| Trigger                                     | Action                                                                       |
| ------------------------------------------- | ---------------------------------------------------------------------------- |
| Encounter unfamiliar tool/framework/library | `skillnet search "<name>"` → download → read SKILL.md → extract useful parts |
| User provides a GitHub URL                  | `skillnet create --github <url>` → evaluate → read SKILL.md → apply          |
| User shares a PDF/DOCX/PPT                  | `skillnet create --office <file>` → evaluate → read SKILL.md → apply         |
| User provides execution logs or data        | `skillnet create <file>` → evaluate → read SKILL.md → apply                  |
| Task hits a wall, no idea how to proceed    | `skillnet search "<problem>" --mode vector` → check results                  |

**Pragmatic note**: In-task triggers should not interrupt flow. If you're in the middle of producing output, finish the current step first, then trigger the search/create. If the task is time-sensitive and you already have a working approach, a search can run in parallel or be deferred to post-task.

## Completion Hook

After completing a significant task, consider: was this difficult, reusable, and high-quality enough to preserve?

If at least two are true — (1) the solution required non-obvious work, (2) it's clearly reusable, (3) the user asked to preserve it — ask for a key and create:

> 这次任务产出了有价值的可复用经验。我想用 SkillNet 的 create 把它沉淀成技能，方便以后一键复用。需要一个 OpenAI 兼容的 API_KEY（可选 BASE_URL 和模型名称，默认 gpt-4o）。只用于本次 create/evaluate，不会外传。可以现在提供吗？

```bash
export API_KEY="<value>"
# If user provided BASE_URL and model:
export BASE_URL="<value>"
skillnet create --prompt "A skill that teaches: [lesson]. Use when: [triggers]. Key steps: [solution]" \
  --output-dir ~/.openclaw/skills --model <model-name>
skillnet evaluate ~/.openclaw/skills/<new-skill> --model <model-name>
```

If user declines → acknowledge and move on.

---

## Search Reference

```bash
# Keyword mode (1–2 short words, fast exact match)
skillnet search "docker" --limit 10
skillnet search "helm chart" --limit 5 --min-stars 3
skillnet search "agent" --category "ai-agent-building"

# Vector mode (longer natural-language queries OK)
skillnet search "how to test React components" --mode vector --threshold 0.7
```

Categories: ai-agent-building, ai-audio-speech, bioinformatics-compbio, cheminformatics-drug-design, cloud-infrastructure-iac, code-quality-refactoring, data-science-visualization, database-design-management, devops-cicd-pipeline, e2e-browser-testing, frontend-ui-engineering, git-workflow-collaboration, llm-app-development, ml-model-pipeline, mobile-cross-platform, prompt-engineering-optimization, react-nextjs-fullstack, rust-systems-programming, security-audit-compliance, technical-documentation, typescript-node-backend.

Fallback: keyword returns 0 → try broader/simpler keywords → then `--mode vector --threshold 0.65`.

## Environment Variables

| Variable       | Needed for                                | Default                     |
| -------------- | ----------------------------------------- | --------------------------- |
| `API_KEY`      | create, evaluate, analyze                 | —                           |
| `BASE_URL`     | custom LLM endpoint                       | `https://api.openai.com/v1` |
| `GITHUB_TOKEN` | download, create --github (private repos) | — (60 req/hr without)       |

**Important**: `search` and `download` (public repos) are **completely free** — no API key, no token. Never ask for credentials before searching.

**Just-in-time key requests** — ask when actually needed, after you've shown the user some value:

- Before `create`/`evaluate`/`analyze`: check `API_KEY`. If empty → ask:

  > **中文**: 我想用 SkillNet 的 create 把这次方案沉淀成一个可复用技能，方便以后一键复用与分享。需要一个 OpenAI 兼容的 API_KEY（可选 BASE_URL 和模型名称，默认 gpt-4o）。只会用于本次 create/evaluate，不会外传。可以现在提供吗？
  >
  > **English**: I'd like to use SkillNet create to package our solution as a reusable skill. This needs an OpenAI-compatible API_KEY (optionally BASE_URL and model name, default gpt-4o). It will only be used for create/evaluate in this run. May I proceed with your key now?

  Then `export API_KEY="<value>"` (and `export BASE_URL` + `--model` if provided) and proceed.
  If user declines → continue the main task.

- If `download` or `create --github` fails (403/rate-limit): check `GITHUB_TOKEN`. If empty → ask:
  > "The download hit a GitHub rate limit. A Personal Access Token (classic, repo scope) would resolve this. Could you provide one?"
  > Then `export GITHUB_TOKEN="<value>"` and retry.
  > If user declines → proceed without that skill.

Never tell user to set env vars themselves — ask for the value, export it, proceed. Never block the main task on a missing key.

---

## Example: Complete Workflow

**Scenario**: User asks "Help me set up a multi-agent system with LangGraph — one agent searches, one codes, one reviews."

**Step 1 — Pre-Task Search (30s):**

```bash
skillnet search "langgraph multi agent" --limit 5
# → 0 results

skillnet search "langgraph supervisor agent" --mode vector --threshold 0.65
# → Found: "langgraph-supervisor-template" (★3, related but generic supervisor pattern)
```

**Step 2 — Download & Selective Apply:**

```bash
skillnet download "https://github.com/.../langgraph-supervisor-template" -d ~/.openclaw/skills
cat ~/.openclaw/skills/langgraph-supervisor-template/SKILL.md
# → Useful: supervisor routing pattern, state schema design, tool-calling conventions
# → Not useful: generic example agents (we need "search→code→review" specifically)
```

**Apply selectively:** Adopt the supervisor routing pattern and state schema from the skill. Build the three specialized agents (searcher, coder, reviewer) from scratch since the skill's generic agents don't fit.

**In-Task Trigger — User also provides a GitHub URL:**

User says: "Also reference https://github.com/langchain-ai/langgraph for the latest API."

```bash
# API_KEY needed → ask user (they've already seen initial progress, so this is non-disruptive)
skillnet create --github https://github.com/langchain-ai/langgraph --output-dir ~/.openclaw/skills
skillnet evaluate ~/.openclaw/skills/langgraph
cat ~/.openclaw/skills/langgraph/SKILL.md
# → Now have detailed API patterns to improve the implementation
```

**Post-Task — Knowledge capture:**

The "search→code→review" pipeline required non-obvious routing logic (conditional edges, retry on review failure). Worth preserving:

```bash
skillnet create --prompt "Multi-agent code pipeline with LangGraph: searcher→coder→reviewer \
  with conditional retry routing when review fails. Use when: building multi-agent code generation \
  systems. Key: use Command for dynamic routing, separate state channels per agent." \
  --output-dir ~/.openclaw/skills
skillnet evaluate ~/.openclaw/skills/langgraph-code-pipeline
# → Safety: Good, Completeness: Good, Executability: Average — acceptable
```

---

## Notes

- Search is free — no API key, no rate limit.
- `skillnet create` outputs a standard skill directory with SKILL.md — no post-processing needed.
- For CLI flags, REST API, and Python SDK reference, see `{baseDir}/references/api-reference.md`.
- For workflow patterns and decision recipes, see `{baseDir}/references/workflow-patterns.md`.
