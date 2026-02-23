# ScienceWorld Environment: Action Reference

## Core Inventory & Object Manipulation Actions
Based on the provided trajectory, the environment accepts the following action patterns for inventory management:

1.  `pick up OBJ [containing STATE] in LOC`
    - **Purpose:** Transfers an object from a location into the agent's inventory.
    - **Examples:**
        - `pick up metal pot containing nothing in kitchen`
        - `pick up battery in workshop`

2.  `move OBJ to OBJ`
    - **Purpose:** Moves an object (from inventory or the current room) into a container or to a location.
    - **Examples:**
        - `move metal pot to blue box`
        - `move battery to table`

## Important Grammar Notes
- **Object Specifiers:** Objects are often described with their state, such as "containing nothing." Including this state in the `pick up` action may be necessary for the action to succeed.
- **Location Specifiers:** The `in LOC` part of the `pick up` action seems to require the room name where the object is located.
- **Container Names:** Containers are referred to by their descriptive names (e.g., `blue box`, `orange box`, `table`, `sink`).

## Related Supporting Actions
- `look around`: Describes the current room and lists visible objects and containers.
- `examine OBJ`: Provides a detailed description of a specific object.
- `look at OBJ`: Describes a container's contents.
- `teleport to LOC`: Instantly moves the agent to a named room.

## Inventory Constraints
From the observed trajectory, the agent's inventory appears to hold one object at a time. The action `drop OBJ` (or its equivalent `move OBJ to LOC`) is used to place the held item back into the environment.
