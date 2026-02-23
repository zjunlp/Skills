---
name: alfworld-receptacle-closer
description: Closes an open receptacle to maintain environment tidiness after inspection. Trigger this after you have finished searching a container and no longer need it open. It helps prevent clutter and is often used as a cleanup step following object searching inside drawers or similar containers.
---
# Skill: Receptacle Closer

## When to Use
Use this skill **immediately after** you have finished inspecting the contents of an open receptacle (e.g., a drawer, cabinet, or container) and have determined you no longer need it to remain open. This is a cleanup action to maintain a tidy environment.

## Core Instruction
1.  **Identify the Target:** The target is the receptacle you just finished searching. It must be open.
2.  **Execute Action:** Perform the `close` action on the identified receptacle.
3.  **Format:** Your output must follow the standard action format: `Action: close {recep}`

## Example from Trajectory
*   **Context:** After opening and inspecting `drawer 2` (finding it empty), the agent closed it.
*   **Thought:** "I should close drawer 2 to keep the room tidy and continue my search."
*   **Action:** `close drawer 2`

## Key Principle
This is a low-cognitive-load, procedural step. Do not overthink it. If you opened a receptacle to look inside and are done with it, close it.
