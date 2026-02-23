---
name: alfworld-object-heater
description: This skill heats a specified object using an available heating appliance (e.g., microwave, stove). Activate this skill when the agent has an object that requires heating and the appliance is prepared. It requires the object and appliance as inputs and results in the object being heated.
---
# Skill: Heat Object

## Purpose
Use this skill to heat a specified object (e.g., potato, soup) using a compatible heating appliance (e.g., microwave, stove) in the ALFWorld environment. The skill handles the sequence of navigation, preparation, and operation of the appliance.

## Prerequisites
1. **Object in Inventory:** The target object must be in the agent's possession (e.g., recently taken from a receptacle).
2. **Appliance Available:** A compatible heating appliance (microwave, stoveburner) must be present and accessible in the environment.
3. **Appliance State:** The appliance must be in a state ready for heating (e.g., microwave door open, stove burner available).

## Core Workflow
Follow this high-level sequence. For detailed, error-prone steps (like checking appliance state), use the bundled script.

1.  **Navigate** to the target heating appliance.
2.  **Prepare** the appliance (e.g., open the microwave door).
3.  **Execute** the heating action on the object using the appliance.
4.  **(Optional) Navigate** to a target location (e.g., dining table) to place the heated object.

## Key Decision Points
*   **Appliance Selection:** Choose the most appropriate appliance based on the object and context (e.g., use a microwave for quick heating, a stove for cooking).
*   **Error Recovery:** If an action fails ("Nothing happened"), re-evaluate the appliance's state (is it open? is it free?) or your inventory (do you have the object?).
*   **Post-Heating Action:** Decide if the heated object should be placed elsewhere (e.g., on a dining table) as part of the task.

## Important Notes
*   The `heat` action requires the object to be in your inventory and the appliance to be ready.
*   Always verify the observation after each action to confirm success before proceeding.
*   Refer to the bundled reference for a detailed breakdown of the example trajectory and common pitfalls.
