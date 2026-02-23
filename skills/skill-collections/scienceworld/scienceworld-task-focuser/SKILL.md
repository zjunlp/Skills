---
name: scienceworld-task-focuser
description: This skill directs the agent's attention to a specific object, either in the environment or inventory, to signal intent or prepare for an operation. It should be triggered before performing a critical action on an object, such as measuring or using it, to ensure the agent is properly aligned with the task. The skill outputs a confirmation of focus, which often is required for subsequent interactive steps in experimental procedures.
---
# Skill: scienceworld-task-focuser

## Purpose
Use this skill to formally declare your intent to interact with a specific object. This is a crucial preparatory step in scientific workflows within the ScienceWorld environment. Focusing on an object signals to the system that subsequent actions (like `use`, `measure`, `connect`) are targeted and deliberate, often unlocking necessary state changes or measurement capabilities.

## When to Use
*   **Before Measurement:** Before using a tool (e.g., a thermometer) on a target object.
*   **Before Complex Operations:** Prior to connecting, mixing, or using one object on another.
*   **Task Alignment:** When a task description explicitly instructs you to "focus on" an object.
*   **Inventory Management:** To confirm you have the correct item in your inventory before proceeding.

## Core Instruction
1.  **Identify the Target Object:** Determine the exact name of the object you need to focus on (e.g., `thermometer`, `metal fork`).
2.  **Locate the Object:** The object may be in the environment (`on table`) or in your inventory (`in inventory`). Use `look around` and `examine` to find it.
3.  **Execute the Focus Action:** Use the exact action format:
    *   If the object is in the environment: `focus on <OBJECT_NAME>`
    *   If the object is in your inventory: `focus on <OBJECT_NAME> in inventory`
4.  **Verify Confirmation:** The environment will respond with "You focus on the `<OBJECT_NAME>`." This is your confirmation to proceed.

## Example from Trajectory
**Scenario:** Measuring the temperature of a metal fork.
1.  **Focus on the tool:** `focus on thermometer in inventory` (Confirms intent to use the thermometer).
2.  **Focus on the target:** `focus on metal fork in inventory` (Confirms the object to be measured).

## Important Notes
*   This skill does not perform the main action (e.g., measuring). It only prepares for it.
*   The `focus` action has no physical effect on the object; it is a declarative meta-action.
*   Always wait for the confirmation observation before attempting the next step in your procedure.
*   If an object is not found, revisit your search logic before attempting to focus.
