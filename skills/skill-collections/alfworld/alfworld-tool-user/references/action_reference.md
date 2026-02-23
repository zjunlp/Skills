# Action Reference for Tool-Based Tasks

## Core Action Templates
The following action formats are available for tool-based interactions in the ALFWorld environment:

1.  `use {tool}` - Apply a tool to a held object or the immediate environment.
2.  `clean {obj} with {tool}`
3.  `heat {obj} with {tool}`
4.  `cool {obj} with {tool}`
5.  `toggle {obj} {recep}` (for tools integrated into receptacles)

## Skill-Specific Mapping
| Task Goal                      | Target Object | Tool          | Appropriate Final Action |
| :----------------------------- | :------------ | :------------ | :----------------------- |
| "examine the X with the Y"     | X (e.g., pillow) | Y (e.g., desklamp) | `use {tool}`             |
| "clean the X with the Y"       | X             | Y             | `clean {obj} with {tool}`|
| "heat the X with the Y"        | X             | Y             | `heat {obj} with {tool}` |
| "cool the X with the Y"        | X             | Y             | `cool {obj} with {tool}` |

## Prerequisite Actions
Before the final tool interaction, the following navigation and manipulation actions are typically required:
- `go to {recep}` - Move to the location of the object or tool.
- `take {obj} from {recep}` - Acquire the target object.
- `open {recep}` - May be needed to access a tool or object.

## Error Signal
- **"Nothing happened"**: Indicates an invalid action. Common causes:
    - Tool or object not in agent's inventory.
    - Not at the correct location.
    - Using the wrong interaction verb for the task.
