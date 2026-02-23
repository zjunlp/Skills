---
name: no-trajectory-analyzer
description: When the execution trajectory is empty or missing, this skill provides a fallback analysis to identify potential missing data, suggest common skill patterns, and generate placeholder metadata for review.
---
# Instructions

This skill activates when the `trajectory` array provided to the agent is empty (`[]`). Its purpose is to handle this specific error state by analyzing the context to understand why the trajectory is missing and to generate actionable, structured output for the user or a downstream process.

## 1. Analyze the Request Context
Examine the user's request and the broader conversation context to hypothesize why a trajectory might be missing. Consider these common scenarios:
*   **New Skill Request:** The user is asking for a brand new skill to be created from a description.
*   **Data Error:** The trajectory data failed to load or was corrupted.
*   **Ambiguous Instruction:** The user's request was too vague to generate a concrete execution path.
*   **Tool Failure:** A previous step that should have produced a trajectory did not execute correctly.

**Output a brief, clear hypothesis** as the first part of your response.

## 2. Generate Diagnostic & Suggestions
Based on your hypothesis, provide structured suggestions. Use the bundled `generate_placeholder.py` script to ensure consistent formatting.
1.  **Run the script** with the skill name and description from the request (or use defaults if not provided).
2.  **Integrate the script's output** (a placeholder skill structure) into your response.
3.  **Append a "Next Steps" section** with 2-3 specific, actionable recommendations for the user (e.g., "Please provide an example execution trajectory," "Clarify the skill's primary action," "Verify the data source for the trajectory.").

## 3. Present Final Output
Structure your final response as follows:
