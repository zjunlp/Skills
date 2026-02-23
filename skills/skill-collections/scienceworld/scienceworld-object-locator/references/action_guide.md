# ScienceWorld Action Guide for Object Location

## Core Actions for Search

### `teleport to <ROOM_NAME>`
- **Purpose:** Instantly move the agent to a specified room.
- **Usage:** `teleport to kitchen`
- **Note:** This is the primary movement action. Use it to navigate between candidate rooms efficiently.

### `look around`
- **Purpose:** Describes the current room, listing all visible objects and containers.
- **Output Format:** "This room is called <ROOM_NAME>. In it, you see: ..."
- **Critical Step:** Always perform this immediately after teleporting to a new room. The object list is in the observation text.

### `examine <OBJECT>`
- **Purpose:** Provides a detailed description of a specific object.
- **Usage:** `examine metal fork`
- **When to Use:** To confirm the identity of an object if its name in the `look around` output is ambiguous or if you need specific properties (e.g., "Is this the *metal* fork or a *plastic* fork?").

### `pick up <OBJECT>`
- **Purpose:** Moves an object from the environment into the agent's inventory.
- **Usage:** `pick up thermometer`
- **When to Use:** If the goal is to acquire and use the object, not just note its location. Perform this after confirming the object's identity.

## Parsing `look around` Observations
The observation text after `look around` is key. Objects are listed after "you see:".
- **Example:** `a thermometer, currently reading a temperature of 10 degrees celsius`
- **Strategy:** Check if the target object's name appears as a separate line item in this list. Simple string matching (e.g., `"thermometer" in observation`) is often sufficient.

## Room Connectivity
Rooms are connected by doors, but the `teleport` action makes physical navigation unnecessary. Focus on the search plan, not door states.

## Inventory Management
- Use `focus on <OBJECT> in inventory` to signal intent on an object you are carrying.
- The trajectory shows that objects in inventory can be used on each other (e.g., `use thermometer on metal fork`).
