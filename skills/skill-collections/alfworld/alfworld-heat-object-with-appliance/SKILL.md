---
name: alfworld-heat-object-with-appliance
description: This skill uses a heating appliance (e.g., microwave, stove) to apply heat to a specified object. It should be triggered when the task requires warming or cooking an item, and the appliance is available and properly prepared (e.g., empty, closed). The input is the object name and appliance name, and the output is the object being heated, changing its state to fulfill task goals like 'heat some egg'.
---
# Skill: Heat Object with Appliance

## Purpose
Heat a specified object using a designated kitchen appliance to change its state (e.g., from cold to hot, raw to cooked) as required by a task.

## Core Workflow
Follow this sequence of actions. Use the bundled `validate_and_plan.py` script to check for common preconditions before starting.

1.  **Navigate & Locate:** Go to the location of the target object.
2.  **Acquire Object:** Take the object from its current receptacle.
3.  **Navigate to Appliance:** Go to the specified heating appliance.
4.  **Prepare Appliance:**
    *   If the appliance is closed, open it.
    *   If the appliance contains other items, you may need to remove them (context-dependent). The default action is to close it and proceed, as some environments abstract this step.
5.  **Execute Heating:** Use the `heat {object} with {appliance}` action.
6.  **Complete Task:** Navigate to the final target location (e.g., a dining table) and put the heated object there.

## Key Decision Points
*   **Appliance State:** If the appliance is occupied, assess if the task implies clearing it. The trajectory shows proceeding without removal is sometimes valid.
*   **Object Location:** The object may not be at the initial location. Be prepared to search other receptacles (e.g., fridge, countertop, cabinet) if not found.
*   **Alternative Appliances:** If the primary appliance (e.g., microwave) is unavailable or broken, consider alternatives like a stoveburner.

## Bundled Resources
*   `scripts/validate_and_plan.py`: A utility to check the initial environment state against the skill's prerequisites.
*   `references/common_heating_appliances.md`: A list of typical appliances and their properties in the ALFWorld environment.
