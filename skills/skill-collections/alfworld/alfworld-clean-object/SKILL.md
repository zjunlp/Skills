---
name: alfworld-clean-object
description: This skill cleans a specified object using an appropriate cleaning receptacle (e.g., sink). It should be triggered when a task requires an object to be in a clean state (e.g., 'clean potato') before proceeding. The skill involves navigating to the cleaning location and performing the clean action, outputting confirmation that the object is now clean.
---
# Instructions

## When to Use
Use this skill when your task requires an object to be in a "clean" state before proceeding (e.g., "clean potato", "wash apple"). The skill is specifically designed for the ALFWorld environment.

## Prerequisites
1. You must already possess the target object in your inventory.
2. You must have identified an appropriate cleaning receptacle (typically a sinkbasin).

## Core Procedure
1. **Navigate to Cleaning Location**: Go to the identified cleaning receptacle (e.g., `sinkbasin 1`).
2. **Execute Clean Action**: Perform the `clean` action using the format: `clean {object_name} with {receptacle_name}`.
3. **Confirm Success**: Verify the environment's observation confirms the object has been cleaned.

## Action Format
- Use the exact ALFWorld action: `clean {obj} with {recep}`
- Example: `clean potato 1 with sinkbasin 1`

## Error Handling
- If the action fails (e.g., "Nothing happened"), consult the troubleshooting guide in the references.
- Ensure you are at the correct receptacle and the object is in your inventory before attempting to clean.

## Post-Condition
After successful execution, the object will be in a clean state. You may proceed with the next step of your task (e.g., placing the clean object in a microwave).
