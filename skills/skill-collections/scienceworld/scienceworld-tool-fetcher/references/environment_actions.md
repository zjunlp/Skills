# ScienceWorld Environment: Action Reference

## Core Actions for Tool Fetching
| Action | Format | Purpose | Notes for Tool Fetcher |
|--------|--------|---------|------------------------|
| `teleport to` | `teleport to LOC` | Move instantly to another room. | Use to navigate to likely tool locations (e.g., `kitchen`, `workshop`). |
| `look around` | `look around` | Get a full description of the current room and visible objects. | **Primary search action.** Use first in a new room. |
| `look at` | `look at OBJ` | Inspect a container to see its contents. | Use if `look around` mentions a container (e.g., `fridge`, `cupboard`) that might hold the tool. |
| `pick up` | `pick up OBJ` | Move a portable object to your inventory. | **Final action.** Use when the tool is directly visible or confirmed inside an open container. |

## Important Environment Assumptions
1.  **Open Containers:** All doors, lids, and drawers are already open. The `open` and `close` actions are **not needed** for this skill.
2.  **Inventory System:** Successful `pick up` moves the object to an implicit inventory. You do not need to manage inventory slots.
3.  **Object Names:** Use the exact name as it appears in observations (e.g., "thermometer", not "the thermometer").
4.  **Room Connectivity:** You can `teleport` directly to any named room.

## Common Tool Locations
*   **Thermometer, Stopwatch, Lighter:** Often found in the `kitchen` or `workshop`.
*   **Specialized Instruments:** Check `workshop`, `art studio`, or `greenhouse`.
*   **Containers to Check:** `fridge`, `cupboard`, `counter`, `drawer`, `shelf`.

## Error Handling Notes
*   If `pick up` fails (e.g., "You can't pick that up"), the object may be fixed or too large. Report this failure.
*   If the tool is not found after searching likely rooms, inform the user.
