---
name: scienceworld-inventory-focus
description: This skill focuses on a specific item within the agent's inventory to confirm its identity or prepare it for use. It should be triggered before using an inventory item in an experiment or when verifying that the correct item has been collected. The skill helps ensure operational readiness and intentional task progression.
---
# Skill: Inventory Focus

## Purpose
Use this skill to intentionally examine and prepare an item from your inventory before using it in a task. This act of "focusing" confirms the item's presence, state, and readiness, reducing errors in multi-step procedures.

## Primary Trigger
Trigger this skill when you have picked up a required item and need to:
1.  Verify it is the correct object for the next step.
2.  Signal your intent to use it.
3.  Prepare it for a subsequent action (e.g., measurement, combination).

## Core Instruction
Execute the `focus on [ITEM] in inventory` action. Replace `[ITEM]` with the exact name of the object you have collected.

## Standard Operating Procedure (SOP)
Follow this sequence when an item from your inventory is needed for a critical task:

1.  **Acquire:** First, ensure the target item is in your inventory (e.g., `pick up thermometer`).
2.  **Focus:** Execute this skill: `focus on [ITEM] in inventory`.
3.  **Proceed:** After receiving confirmation, proceed with the intended use of the item (e.g., `use thermometer on unknown substance B`).

## Example from Trajectory
*   **Goal:** Measure the temperature of `unknown substance B`.
*   **Procedure:**
    1.  `pick up thermometer`
    2.  `focus on thermometer in inventory` *(This skill)*
    3.  `pick up unknown substance B`
    4.  `focus on unknown substance B in inventory` *(This skill)*
    5.  `use thermometer on unknown substance B`

## Key Principle
Treat `focus on` as a deliberate checkpoint. It does not change the state of the object but changes the agent's state of awareness and intent, leading to more reliable task execution.
