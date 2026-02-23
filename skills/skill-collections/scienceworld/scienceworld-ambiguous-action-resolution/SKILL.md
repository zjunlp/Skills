---
name: scienceworld-ambiguous-action-resolution
description: Resolves system ambiguity prompts by selecting the appropriate action from numbered options. Trigger this skill when the environment presents multiple identical action possibilities and requires explicit selection. This ensures task progression when the system cannot automatically disambiguate identical object instances.
---
# Skill: Ambiguous Action Resolution

## When to Use
Activate this skill **only** when the environment returns an "Ambiguous request" observation with a numbered list of identical action options. This typically occurs when multiple identical object instances exist (e.g., five identical banana seeds in a jar) and the system cannot determine which specific instance you intend to act upon.

## Core Procedure
1.  **Identify the Ambiguity:** Recognize the prompt format: `"Ambiguous request: Please enter the number for the action you intended (or blank to cancel):"` followed by a numbered list (0, 1, 2...).
2.  **Parse the Options:** Quickly scan the listed options. They will be functionally identical but refer to different instances of the same object.
3.  **Select a Number:** Choose the **lowest available number** (typically `0`) to proceed. The specific instance is irrelevant for task completion; any valid selection will satisfy the system's requirement and allow the action to execute.
4.  **Execute:** Output the selected number as the next action (e.g., `Action: 0`).

## Key Principles
*   **Efficiency:** Do not overthink. The objects are identical; any choice is valid.
*   **Consistency:** Always default to the first option (`0`) unless a previous step in the task logic specifically requires targeting a different instance (which is rare).
*   **Focus:** This skill is a **disambiguation mechanic**, not a decision-making process. Its sole purpose is to bypass a system prompt blocking progress.

## Example from Trajectory
**Observation:** "Ambiguous request: Please enter the number for the action you intended (or blank to cancel): 0: move banana seed (in seed jar, in inventory, in agent, in greenhouse) to flower pot 1 (in greenhouse) 1: move banana seed (in seed jar, in inventory, in agent, in greenhouse) to flower pot 1 (in greenhouse) ..."
**Correct Skill Application:** `Action: 0`

## Anti-Patterns to Avoid
*   Do NOT use this skill for non-ambiguous choices or menu selections where options have different meanings.
*   Do NOT trigger this skill if the observation does not contain the exact "Ambiguous request" phrase.
*   Do NOT waste steps analyzing the differences between identical instances.
