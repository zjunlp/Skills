# Action Primitives for Recipe Acquisition

## Core Actions Used
1. **look around** - Surveys the current room and lists all visible objects
2. **pick up <object>** - Moves an object from the environment to inventory

## Related Supporting Actions
- **examine <object>** - Get detailed description of an object (use if uncertain about recipe identity)
- **read <object>** - Read contents of a document (use after acquisition)
- **teleport to <location>** - Move to different rooms to search for recipe

## Action Constraints
- All containers are already open, so no need to `open` before accessing contents
- Recipes are typically standalone objects, not inside containers
- The `pick up` action should work directly on visible recipe objects

## Common Observation Patterns
When `look around` reveals a recipe, it typically appears in formats like:
- "A recipe titled [name]"
- "instructions to make [substance]"
- "a manual for [process]"

## Error Messages to Handle
- "You cannot pick that up" - May indicate the object isn't actually a recipe
- "You don't see that" - Recipe may not be in current room
- "Your inventory is full" - Need to manage inventory space first
