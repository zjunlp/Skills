---
name: alfworld-tool-user
description: This skill uses a tool on a target object to accomplish a specific interaction, such as examining, cleaning, or manipulating. It should be triggered when both the object and the required tool are in the agent's possession or within reach. The skill takes the tool and object as implicit inputs and executes the appropriate interaction action (e.g., 'use', 'clean', 'heat') to progress the task.
---
# Instructions
1.  **Objective:** Use a specified tool on a target object to complete a task (e.g., examine, clean, heat).
2.  **Prerequisites:** The agent must have the target object in its inventory or be at its location. The required tool must be accessible (in inventory or on a nearby receptacle).
3.  **Procedure:**
    a.  **Locate Target:** Navigate to and acquire the target object if not already held.
    b.  **Locate Tool:** Navigate to and identify the required tool.
    c.  **Execute Interaction:** Perform the environment-specific action to apply the tool to the object (e.g., `use {tool}`, `clean {obj} with {tool}`). The exact action verb is determined by the task context.
4.  **Error Handling:** If an action fails (e.g., "Nothing happened"), reassess the object/tool state and location before retrying or attempting an alternative approach.
