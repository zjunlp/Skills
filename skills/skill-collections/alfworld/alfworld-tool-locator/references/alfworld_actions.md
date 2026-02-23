# ALFWorld Action Reference

## Available Actions
The ALFWorld environment supports the following action types:

### Navigation & Interaction
1. `go to {recep}` - Move to a receptacle
2. `take {obj} from {recep}` - Pick up an object from a receptacle
3. `put {obj} in/on {recep}` - Place an object in/on a receptacle
4. `open {recep}` - Open a receptacle
5. `close {recep}` - Close a receptacle
6. `toggle {obj} {recep}` - Toggle an object (e.g., switch)
7. `clean {obj} with {recep}` - Clean an object using a receptacle
8. `heat {obj} with {recep}` - Heat an object using a receptacle
9. `cool {obj} with {recep}` - Cool an object using a receptacle

## Observation Format
After each action, the environment returns an observation:
- Success: "You [action] the [object]"
- Failure: "Nothing happened"
- Description: "On the [recep], you see [object1], [object2], ..."

## Common Receptacle Types
- **Surfaces**: sidetable, countertop, table, shelf, desk
- **Storage**: drawer, cabinet, fridge, microwave
- **Furniture**: sofa, armchair, bed, toilet
- **Containers**: garbagecan, box, basket

## Tool Naming Conventions
Tools in ALFWorld typically follow these naming patterns:
- `{toolname} {number}` - e.g., "desklamp 1", "knife 2"
- Common tools: desklamp, knife, spoon, book, remotecontrol, cellphone
- Cleaning tools: sponge, cloth, towel

## Search Strategy Guidelines
1. **Prioritize likely locations**: Tools are often found on surfaces near their use context
2. **Check systematically**: Don't skip receptacles
3. **Handle failures gracefully**: If "Nothing happened", try different approach
4. **Use observations**: The observation tells you what's visible at a location

## Example Task Flow
