# ScienceWorld Action Primer for Room Scanning

## The `look around` Action
*   **Syntax:** `look around`
*   **Effect:** Returns a detailed textual description of the agent's current room.
*   **Key Observation Fields:**
    1.  `This room is called <ROOM_NAME>.`
    2.  `In it, you see:` - Lists all objects, substances, and the agent itself.
    3.  `On <CONTAINER> is:` / `In <CONTAINER> is:` - Describes contents of open containers and furniture.
    4.  `You also see:` - Lists all doorways to adjacent rooms.

## Related Contextual Actions
Use information from `look around` to inform these subsequent actions:

| Action | Syntax Example | Use Case After Scanning |
| :--- | :--- | :--- |
| `examine` | `examine thermometer` | Get detailed properties of a specific object seen in the room. |
| `look at` | `look at cupboard` | See the contents of a specific container (more focused than `look around`). |
| `teleport to` | `teleport to kitchen` | Move to a room listed in the "You also see:" connections. |
| `pick up` | `pick up thermometer` | Take an object that is listed as visible and accessible. |

## Common Room Types & Key Objects
Based on the trajectory, here are typical rooms and objects of interest:
*   **Kitchen:** `thermometer`, `oven`, `stove`, `metal pot`, various `boxes`, `cupboard`.
*   **Foundry:** `blast furnace` (for high-temperature heating).
*   **Hallway:** Central hub with doors to many other rooms.
*   **Workshop/Greenhouse/Art Studio:** Likely contain specialized tools or materials.

## Best Practice: The Scan-Act Cycle
1.  **SCAN:** Use `look around` upon entry or when uncertain.
2.  **PARSE:** Identify the objects relevant to your current goal from the description.
3.  **ACT:** Execute a targeted action (`pick up`, `examine`, `use`).
4.  **RE-SCAN (if needed):** If your action changes the room state significantly (e.g., activating a furnace, moving many items), consider another `look around` to confirm the new state.
