# Action Primer for ScienceWorld Object Location

## Core Action for This Skill
- `look around`: Describes the current room, listing all visible objects and their states. This is the primary action for environmental awareness.

## Related Actions (For Subsequent Steps)
- `pick up OBJ`: Moves a found object from the room to the agent's inventory.
- `examine OBJ`: Provides a detailed description of a specific object (useful for verification).
- `teleport to LOC`: Moves the agent to a different room to continue the search.

## Observation Format
A typical `look around` observation has this structure:
