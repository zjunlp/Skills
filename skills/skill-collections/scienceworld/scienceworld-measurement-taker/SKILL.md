---
name: scienceworld-measurement-taker
description: This skill uses a measurement tool (like a thermometer) on a target substance or object to obtain a quantitative reading. It should be triggered when the task requires assessing a property (e.g., temperature) to make a conditional decision. The skill outputs the measured value, which determines subsequent actions such as classification or placement.
---
# Measurement Taker Skill

## Purpose
Use this skill when you need to measure a quantitative property (e.g., temperature, weight, pH) of a target object or substance to inform a subsequent decision or action.

## Core Workflow
1.  **Identify & Acquire Tool:** Locate and obtain the correct measurement tool (e.g., thermometer, scale).
2.  **Identify & Acquire Target:** Locate and obtain the target object or substance to be measured.
3.  **Prepare Measurement Environment:** Move to a location suitable for the measurement and any required follow-up actions (e.g., near classification bins).
4.  **Execute Measurement:** Use the tool on the target to obtain the numerical reading.
5.  **Interpret & Act:** Based on the measured value and the task's conditional logic, execute the appropriate follow-up action (e.g., place the target in a specific container).

## Key Principles
*   **Tool First:** Secure the measurement tool before handling the target, unless the task specifies otherwise.
*   **Environmental Awareness:** Proactively identify where follow-up actions (like placement) will occur and position yourself accordingly before measuring.
*   **Verification:** Use `focus on` actions to confirm you have the correct tool and target in your inventory before proceeding.
*   **Conditional Logic:** The measured value directly determines the next action. Clearly map out the decision thresholds (e.g., "above 100.0", "below 100.0") before measuring.

## Common Actions Sequence
`look around` → `teleport to [Room]` → `pick up [Tool]` → `focus on [Tool]` → `pick up [Target]` → `focus on [Target]` → `teleport to [Action Room]` → `use [Tool] on [Target]` → `[Conditional Action based on reading]`
