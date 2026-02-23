# Common Benchmark Attributes Guide
This reference defines key attributes often requested when comparing academic benchmarks/datasets.

## 1. Tasks
*   **Definition:** The distinct problem types or categories evaluated by the benchmark.
*   **Interpretation:** Can be a count of broad categories (e.g., "5" for KOR-Bench's Operation, Logic, Cipher, Puzzle, Counterfactual) or a count of specific task instances (e.g., "23" for BBH's suite of challenges). Prioritize the categorization used in the benchmark's primary paper.
*   **Source:** Look in the paper's abstract, introduction, or a dedicated "Benchmark Structure" section.

## 2. Trainable
*   **Definition:** Whether the benchmark provides a dedicated dataset intended for model training (fine-tuning), as opposed to being solely for evaluation.
*   **Values:**
    *   `\ding{51}` (Yes): The benchmark includes a training split. The associated paper often discusses fine-tuning experiments.
    *   `\ding{55}` (No): The benchmark is for evaluation only (e.g., via few-shot or zero-shot prompting). Common for "hard" or "challenge" benchmarks (BBH, BBEH).
*   **Source:** Check for mentions of "training set," "fine-tuning," "train/validation/test splits," or data generation procedures intended for training.

## 3. Adjustable Difficulty
*   **Definition:** Whether the benchmark is designed with mechanisms to systematically vary the complexity or difficulty level of its problems.
*   **Values:**
    *   `\ding{51}` (Yes): The benchmark explicitly controls difficulty parameters (e.g., puzzle size, search space, number of logical constraints). Papers may mention "controllable complexity" or "multiple difficulty levels."
    *   `\ding{55}` (No): The benchmark presents problems at a fixed, predefined level of challenge.
*   **Source:** Look for sections on "dataset generation," "complexity measures," or "scaling studies."

## 4. Core Purpose
*   **Definition:** A brief description of what the benchmark aims to measure or probe.
*   **Examples:** "Logical reasoning on constraint satisfaction problems," "Knowledge-orthogonal reasoning," "Extremely challenging general reasoning capabilities."
*   **Source:** The paper's abstract is usually the best source.

## Data Collection Tips
*   **Primary Source First:** Always consult the arXiv paper or official project page (GitHub) as the most authoritative source.
*   **Corroborate:** Use web search results to find community descriptions or blog posts that might clarify ambiguous points, but weight them lower than the primary source.
*   **Inference:** If an attribute is not mentioned, it is often safe to infer its absence (e.g., no mention of a training set implies evaluation-only). State this inference clearly in your final summary.
