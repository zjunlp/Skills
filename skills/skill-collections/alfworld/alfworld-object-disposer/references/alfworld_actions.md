# AlfWorld Action Reference for Disposal

## Relevant Action Template
`put {obj} in/on {recep}`

## Description
Places an object into or onto a receptacle. For disposal tasks, the receptacle is typically a `garbagecan`.

## Preconditions
1. The agent must be holding the object (`{obj}`).
2. The agent must be at the location of the receptacle (`{recep}`).
3. The receptacle must be accessible (e.g., not inside a closed container).

## Common Disposal Receptacles
- `garbagecan`
- `sinkbasin` (for liquid disposal)
- `toilet` (if available in the environment)

## Error Handling
If the action fails and returns "Nothing happened":
1. Verify the object identifier is correct and matches an object in inventory.
2. Verify the receptacle identifier is correct and present.
3. Ensure the receptacle can accept the object (some receptacles may have capacity limits).
