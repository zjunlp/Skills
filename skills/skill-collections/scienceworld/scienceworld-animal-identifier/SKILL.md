---
name: scienceworld-animal-identifier
description: This skill identifies and focuses on specific animals or biological entities present in the environment. It should be triggered when the task requires examining, comparing, or interacting with animals, such as determining lifespan attributes. The skill takes an animal identifier as input and outputs a confirmation of focus, aiding in targeted analysis for scientific experiments or comparisons.
---
# Instructions

## Purpose
This skill enables the agent to locate and focus on a specified animal or biological entity within the ScienceWorld environment. It is designed for tasks involving animal comparison, examination, or interaction, such as determining lifespan extremes.

## Core Workflow
1.  **Locate Target Environment:** If the target animal is known to be in a specific location (e.g., 'outside'), teleport there first.
2.  **Survey the Area:** Use `look around` to list all visible objects and entities in the current location.
3.  **Identify Target:** Parse the observation to find the specified animal identifier (e.g., 'parrot egg', 'baby dragonfly').
4.  **Execute Focus:** Use the `focus on <ANIMAL>` action to signal intent on the identified target object.

## Key Principles
*   **Context-Aware Navigation:** Always verify your current location. If you are not in the correct room for the target animal, use `teleport to <LOCATION>` as the first step.
*   **Precise Targeting:** The `focus on` action requires the exact object name as it appears in the `look around` observation (e.g., "baby dragonfly", not just "dragonfly").
*   **Sequential Execution:** This skill is often used in a sequence (e.g., find longest-lived, then shortest-lived animal). Complete the focus action for one target before proceeding to the next.

## Input/Output
*   **Input:** An animal identifier or a task description implying the need to identify/focus on an animal (e.g., "find the animal with the longest life span").
*   **Output:** A confirmation that the `focus on <ANIMAL>` action has been successfully executed on the correct target.
