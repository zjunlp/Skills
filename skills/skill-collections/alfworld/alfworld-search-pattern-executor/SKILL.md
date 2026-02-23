---
name: alfworld-search-pattern-executor
description: Systematically searches a sequence of likely locations for a target object based on common sense. Takes a list of candidate receptacles, orchestrates navigation and inspection, and outputs when the target is found or all locations are exhausted.
---
# Skill: Systematic Object Search

## Purpose
Use this skill when you need to find a specific object in a household environment without prior knowledge of its exact location. The skill implements a robust search pattern based on common sense about where objects are typically stored.

## Core Workflow
1.  **Input:** A target object name (e.g., `remotecontrol`) and an ordered list of candidate receptacles to search.
2.  **Process:** Navigate to each candidate location in sequence. For each receptacle:
    *   If it's closed, open it.
    *   Inspect its contents.
    *   If the target object is found, take it and proceed to the placement phase.
    *   If the receptacle was opened and is empty, close it before moving on.
3.  **Output:** Success when the object is found, or a failure state after all candidates are exhausted.

## Key Principles
*   **Methodical Search:** Do not skip locations in the provided sequence unless the object is found.
*   **State Management:** Always close drawers/cabinets after checking them if they were opened.
*   **Focus:** Once the object is found, immediately transition to the next phase of the task (e.g., `put`). Avoid redundant searches.
*   **Error Handling:** If an action fails (e.g., "Nothing happened"), the skill logic in `scripts/search_orchestrator.py` provides fallback reasoning.

## Usage Example
**Goal:** "find two remotecontrol and put them in armchair."
**Skill Execution:**
1.  Activate this skill with target=`remotecontrol`, candidates=`['sofa 1', 'sidetable 1', 'dresser 1', 'drawer 1', 'drawer 2', 'drawer 3', 'drawer 4', 'coffeetable 1']`.
2.  The skill searches locations in order. It finds the first `remotecontrol` on `coffeetable 1`.
3.  The agent takes the object and proceeds to place it in `armchair 1`.
4.  The skill is re-activated for the second `remotecontrol`, searching the remaining candidates (or the same list). It finds the second `remotecontrol` on `coffeetable 1`.
5.  The agent takes and places the second object, completing the task.

## Integration
This skill is designed to be called as a subroutine within a larger task plan. The bundled `search_orchestrator.py` script handles the low-level action sequencing to prevent logical errors and repetitive mistakes observed in the learning trajectory.
