# SkillNet Workflow Patterns

Recipes for common scenarios. Each pattern shows the trigger signal, the recommended actions, and the expected outcome. Remember: search and download are free and fast — never hesitate to search.

---

## Pattern 1: "I need a skill for an unfamiliar domain"

**Trigger**: You received a task involving a technology, framework, or domain you lack expertise in.

**Steps**:

1. Identify 2–3 keywords describing the domain (e.g., "kubernetes helm chart")
2. `skillnet search "kubernetes helm" --limit 5`
3. If 0 results → retry: `skillnet search "kubernetes deployment" --mode vector --threshold 0.65`
4. Review results — check evaluation scores (prefer Good Safety + Good Executability)
5. `skillnet download "<top-result-url>" -d ~/.openclaw/skills`
6. Read the downloaded SKILL.md — extract patterns, constraints, and tool choices relevant to your task
7. If the skill only partially matches, use what's useful and fill gaps yourself

**Outcome**: You have domain expertise loaded. Apply selectively — not everything in the skill may fit your exact problem.

---

## Pattern 2: "User wants me to learn a GitHub project"

**Trigger**: User says "understand this repo", "learn this project", or directly provides a GitHub URL.

**Steps**:

1. `skillnet create --github https://github.com/owner/repo --output-dir ~/.openclaw/skills`
2. Wait for creation (analyses README, source structure, key files)
3. `skillnet evaluate ~/.openclaw/skills/<generated-name>`
4. If evaluation shows "Poor" on any dimension → warn the user, suggest manual review
5. Read the generated SKILL.md — now you understand the project's architecture, patterns, and usage

**Outcome**: The GitHub repo is now encoded as a reusable skill.

---

## Pattern 3: "Post-task knowledge capture" (Completion Hook)

**Trigger**: You just completed a significant task **and did NOT use `skillnet create` during it**.

**Decision rule** — enter the create path only if **at least two** are true:

- The solution was genuinely difficult (non-obvious, required substantial investigation)
- The output is clearly reusable — another agent would benefit
- The user explicitly asked to summarize experience or create a skill

Routine tasks, minor fixes, or straightforward work → do NOT create.

**Steps**:

1. If no `API_KEY` → use the standard ask text to request one
2. Formulate what you learned as a concise description
3. **MUST use `skillnet create`** — never manually write a SKILL.md
4. `skillnet create --prompt "A skill that teaches: [lesson learned]. Use when: [trigger conditions]. Key steps: [solution outline]" --output-dir ~/.openclaw/skills`
5. `skillnet evaluate ~/.openclaw/skills/<new-skill>`

**Outcome**: Knowledge is preserved via SkillNet's standardized format. If user declined the key, do NOT hand-write a SKILL.md as workaround — either use `skillnet create` or skip.

---

## Pattern 4: "Plan-phase skill augmentation"

**Trigger**: At the beginning of a multi-step task, during planning.

**Time budget**: Keep the entire search phase under 2 minutes. Don't let skill discovery delay the actual work.

**Steps**:

1. Break the task into sub-domains
2. For each sub-domain, check: "Do I have a local skill for this?"
3. For gaps → `skillnet search "<sub-domain>"` in batch
4. Download the most relevant skills (skip marginal matches)
5. Skim each SKILL.md — extract only the parts that inform your plan
6. Execute the plan with augmented capabilities

**Outcome**: Your plan is informed by domain expertise from the skill library.

---

## Pattern 5: "Clean up my skill library"

**Trigger**: User asks to organize, audit, or clean up their skills. Or you notice the managed skills directory has >30 skills.

**Steps**:

1. `skillnet analyze ~/.openclaw/skills`
2. Review `relationships.json`:
   - `similar_to` pairs → consider merging (keep the one with higher evaluation scores)
   - `depend_on` chains → ensure dependencies are all installed
   - `belong_to` hierarchies → organize into subdirectories if helpful
3. For skills with unknown quality → `skillnet evaluate <skill-path>`
4. Remove or archive skills scoring "Poor" on Safety or multiple "Poor" dimensions

**Outcome**: A lean, high-quality skill library with understood relationships.

---

## Pattern 6: "Create skill from user's document"

**Trigger**: User shares a PDF, PPT, or Word document and wants it encoded as a skill.

**Steps**:

1. Save the document to a local path if not already on disk
2. `skillnet create --office /path/to/document.pdf --output-dir ~/.openclaw/skills`
3. Evaluate the created skill
4. Read SKILL.md to verify the knowledge was correctly extracted

**Outcome**: Domain knowledge from the document is now accessible as a skill.

---

## Decision Matrix: Which SkillNet Feature to Use

| Situation                            | Feature                   | Command                                        |
| ------------------------------------ | ------------------------- | ---------------------------------------------- |
| Need expertise in a new domain       | **search** + **download** | `skillnet search ... && skillnet download ...` |
| User provides a GitHub repo to learn | **create** (github)       | `skillnet create --github <url>`               |
| Finished a complex task with lessons | **create** (prompt)       | `skillnet create --prompt "..."`               |
| User shares a knowledge document     | **create** (office)       | `skillnet create --office <file>`              |
| User provides execution logs or data | **create** (trajectory)   | `skillnet create --trajectory <trajectory-file>` |
| Unsure about a skill's quality       | **evaluate**              | `skillnet evaluate <path-or-url>`              |
| Too many skills, need organization   | **analyze**               | `skillnet analyze <dir>`                       |
