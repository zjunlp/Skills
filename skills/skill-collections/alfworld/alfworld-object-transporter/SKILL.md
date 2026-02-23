---
name: alfworld-object-transporter
description: This skill picks up a target object from its current receptacle and moves it to a specified destination receptacle. It should be triggered when the agent has located an object and needs to relocate it to complete a task (e.g., moving a laptop to a desk). The skill requires the object identifier and source location as input, and it outputs the action sequence to take and transport the object.
---
# Instructions

## When to Use
Use this skill when you have:
1. **Identified the target object** (e.g., `laptop 1`) and its **current source receptacle** (e.g., `bed 2`).
2. **Identified the destination receptacle** (e.g., `desk 1`).
3. The goal requires moving the object to the destination to complete a task.

## Input Requirements
You must provide the following information to execute this skill:
- **`target_object`**: The identifier of the object to move (e.g., `laptop 1`).
- **`source_receptacle`**: The identifier of the receptacle where the object is currently located (e.g., `bed 2`).
- **`destination_receptacle`**: The identifier of the receptacle where the object must be placed (e.g., `desk 1`).

## Execution Flow
1. **Navigate to Source**: Go to the `source_receptacle`.
2. **Pick Up Object**: Take the `target_object` from the `source_receptacle`.
3. **Navigate to Destination**: Go to the `destination_receptacle`.
4. **Place Object**: Put the `target_object` in/on the `destination_receptacle`.

## Action Format
All actions must follow the Alfworld environment's strict format:
- `go to {receptacle}`
- `take {object} from {receptacle}`
- `put {object} in/on {receptacle}`

## Error Handling
- If an action fails (environment returns "Nothing happened"), consult the troubleshooting guide in `references/troubleshooting.md`.
- If the object or receptacle is not found, re-scan the environment before retrying.

## Bundled Resources
- **Script**: `scripts/transport_sequence.py` provides a deterministic sequence generator.
- **Reference**: `references/troubleshooting.md` contains common failure patterns and solutions.
