# AlfWorld Action Guide for Object Placement

## Core Action
`put {obj} in/on {recep}`

## Preposition Rules
Use the following guide to choose `in` or `on`:

### Use `in` for:
- Enclosed containers: `cabinet`, `fridge`, `microwave`, `drawer`, `garbagecan`, `box`, `shelf` (if it has walls).
- Receptacles that hold items internally.

### Use `on` for:
- Surfaces: `countertop`, `desk`, `table`, `shelf` (open).
- Holders: `toiletpaperhanger`, `towelholder`, `handtowelholder`, `plateholder`.
- Appliances: `stoveburner`, `toaster`.

## Common Failure Modes & Solutions
1. **"Nothing happened" on `put` action:**
   - **Cause 1:** The agent is not holding the specified `{obj}`.
     - *Solution:* Use `take {obj} from {source_recep}` first.
   - **Cause 2:** The `{recep}` is closed or inaccessible.
     - *Solution:* Use `open {recep}` before placing the object inside.
   - **Cause 3:** The preposition (`in`/`on`) is incorrect for the receptacle type.
     - *Solution:* Consult the table above and try the alternative preposition.
   - **Cause 4:** The receptacle is already occupied/full.
     - *Solution:* Clear the receptacle first (e.g., remove existing objects).

2. **Object not found for pickup:**
   - Search common locations: countertops, tables, shelves, cabinets, and other receptacles relevant to the object's type (e.g., toilet paper is often near a toilet).

## Example from Trajectory
**Goal:** Put a toiletpaper in toiletpaperhanger.
**Successful Sequence:**
1. `go to toiletpaperhanger 1` (Verify it's empty)
2. `go to toilet 1` (Locate object)
3. `take toiletpaper 1 from toilet 1` (Acquire object)
4. `go to toiletpaperhanger 1` (Navigate to target)
5. `put toiletpaper 1 on toiletpaperhanger 1` (Execute placement - uses 'on' for a holder)
