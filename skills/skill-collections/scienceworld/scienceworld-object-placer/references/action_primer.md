# Action Primer for ScienceWorld Environment

This reference details the actions used in the provided trajectory that are relevant to the object-placer skill's prerequisite steps.

## Core Placement Action
*   `move OBJ to OBJ`: Transfers an object to a container. This is the final action of the skill.
    *   *Example:* `move metal pot to blue box`

## Supporting Actions (Pre-Skill)
These actions are typically part of the assessment phase that occurs *before* this skill is triggered.
*   `pick up OBJ`: Moves an object from the environment into your inventory. Use this if the target object needs to be carried.
    *   *Example:* `pick up metal pot`
*   `look around`: Describes the current room and lists all visible objects and containers. Use this to locate the target object and destination containers.
*   `examine OBJ`: Provides a detailed description of a specific object. Use this to identify object properties or terminals.
*   `focus on OBJ`: Signals intent on a task-relevant object. Can be used to confirm target selection.
*   `connect OBJ to OBJ`: Connects two electrical components. Used in the example trajectory to build a test circuit.
*   `wait1`: Waits for one iteration, allowing time-dependent effects (like a buzzer turning on) to be observed.

## Container Notes
*   Containers like `blue box` and `orange box` are typically empty and ready to receive objects.
*   You can use `look at OBJ` to inspect a container's contents, but this is often unnecessary if the task explicitly states the container is empty.
