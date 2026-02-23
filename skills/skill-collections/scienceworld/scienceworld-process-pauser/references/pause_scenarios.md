# Pause Scenario Reference

This document outlines common scenarios in ScienceWorld tasks where introducing a deliberate pause improves performance, based on trajectory analysis.

## 1. After Creating an Intermediate Product
*   **Trigger:** Observation confirms creation of a new substance or assembly (e.g., "X and Y mix to produce Z").
*   **Purpose:** Evaluate the intermediate result's properties before proceeding.
*   **Example Action:** `wait1`
*   **Trajectory Example:** After `mix jug` produced "orange paint," the agent used `wait1`.

## 2. Before a Critical or Irreversible Step
*   **Trigger:** About to add a final component, activate a device, or perform a mix that completes the task.
*   **Purpose:** Final verification of conditions and proportions to avoid error.
*   **Example Action:** `wait` (longer pause for complex verification)
*   **Trajectory Logic:** Before the final `pour` and `mix` to create red-orange paint.

## 3. After a State Change Observation
*   **Trigger:** Observation begins with "You pour," "You connect," "You activate," etc.
*   **Purpose:** Process the consequence of the action before deciding the next step.
*   **Example Action:** `wait1`
*   **Trajectory Logic:** General best practice observed in successful trajectories.

## 4. When Evaluating an Object
*   **Trigger:** After using `focus on OBJ` or `examine OBJ`.
*   **Purpose:** Deliberately process the descriptive information received.
*   **Example Action:** `wait1`
*   **Trajectory Example:** After `focus on orange paint`.

## Anti-Patterns: When NOT to Pause
*   **During Exploration:** Do not pause after `look around` or `teleport`. Proceed directly to examining objects.
*   **After Simple Fetch Actions:** Do not pause after `pick up OBJ` unless it's a rare or final component.
*   **If the task is time-sensitive:** Some simulated processes may degrade if paused too long. Use context to judge.
