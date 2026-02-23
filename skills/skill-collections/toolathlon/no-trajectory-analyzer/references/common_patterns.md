# Common Skill Patterns for Trajectory Inference

When a trajectory is missing, the requested skill often fits one of these high-level patterns. Use this list to inform suggestions and placeholder generation.

## 1. Data Fetcher & Transformer
*   **Pattern:** Fetch data from a source (API, file, database), apply a transformation (filter, aggregate, convert format), and output results.
*   **Typical Steps:** Validate input → Connect to source → Fetch data → Apply logic/transforms → Format output → Handle errors.
*   **Example Skills:** `api-poller`, `csv-to-json-converter`, `database-snapshot`.

## 2. Content Analyzer & Summarizer
*   **Pattern:** Take a body of text or structured data, analyze it for key information, and produce a summary, extract key points, or answer specific questions.
*   **Typical Steps:** Parse input → Extract entities/themes → Apply analysis rules/LLM call → Synthesize findings → Present summary.
*   **Example Skills:** `document-summarizer`, `sentiment-analyzer`, `meeting-minutes-generator`.

## 3. Orchestrator & Sequencer
*   **Pattern:** Coordinate the execution of other skills or tools in a specific sequence, passing data between steps.
*   **Typical Steps:** Receive parameters → Validate sequence → Execute step 1 → Pass result to step 2 → ... → Aggregate final results.
*   **Example Skills:** `multi-step-data-pipeline`, `validation-workflow`, `deployment-automator`.

## 4. Generator & Creator
*   **Pattern:** Create new content or structured data based on templates, rules, or prompts.
*   **Typical Steps:** Load template/assets → Apply variables/rules → Generate content → Validate output → Deliver.
*   **Example Skills:** `report-generator`, `code-skeleton-builder`, `email-template-filler`.

## 5. Validator & Checker
*   **Pattern:** Check input data or system state against rules, schemas, or expectations and return a pass/fail status with details.
*   **Typical Steps:** Receive target → Load validation rules → Perform checks → Compile violations → Return report.
*   **Example Skills:** `config-validator`, `compliance-checker`, `syntax-linter`.
