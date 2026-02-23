# Common ScienceWorld Task Patterns

This reference catalogs frequent instruction patterns and their corresponding parsing logic.

## Pattern 1: Find and Move
*   **Instruction Template:** "Find a(n) [OBJECT_TYPE]. First, focus on the thing. Then, move it to the [CONTAINER] in the [ROOM]."
*   **Parsed Components:**
    *   Target Object Type: `[OBJECT_TYPE]` (e.g., non-living thing, red wire, light bulb)
    *   Target Location: `[CONTAINER]` in `[ROOM]`
    *   Action Sequence: `find` -> `focus on` -> `move to`
*   **Execution Flow:**
    1.  Teleport to `[ROOM]`.
    2.  `look around`.
    3.  Identify an object matching `[OBJECT_TYPE]`.
    4.  `focus on [OBJECT]`.
    5.  `move [OBJECT] to [CONTAINER]`.

## Pattern 2: Activate/Use Device
*   **Instruction Template:** "Turn on/activate/use the [DEVICE]."
*   **Parsed Components:**
    *   Target Object: `[DEVICE]`
    *   Action: `activate` or `use`
*   **Execution Flow:**
    1.  Locate the `[DEVICE]` (may require room change).
    2.  `activate [DEVICE]` or `use [DEVICE]`.

## Pattern 3: Create Mixture
*   **Instruction Template:** "Mix [SUBSTANCE_A] and [SUBSTANCE_B] in the [CONTAINER]."
*   **Parsed Components:**
    *   Target Objects: `[SUBSTANCE_A]`, `[SUBSTANCE_B]`
    *   Target Location: `[CONTAINER]`
    *   Action: `pour` -> `mix`
*   **Execution Flow:**
    1.  Locate substances and container.
    2.  `pour [SUBSTANCE_A] into [CONTAINER]`.
    3.  `pour [SUBSTANCE_B] into [CONTAINER]`.
    4.  `mix [CONTAINER]`.

## Object Classification Guide
Use this to disambiguate the **Target Object Type**:
*   **Non-living thing:** Any man-made object (wire, bulb, battery, tool, box). Not substances (air, water).
*   **Electrical component:** battery, wire, switch, light bulb.
*   **Container:** box, freezer, beaker, vial.
*   **Substance:** air, water, chemical solutions (often in containers).

## Action Mapping
*   `find` -> `look around` / `examine`
*   `focus on` -> `focus on`
*   `move` -> `move OBJ to OBJ`
*   `turn on/off` -> `activate` / `deactivate`
*   `connect` -> `connect OBJ to OBJ`
*   `pour` -> `pour OBJ into OBJ`
*   `mix` -> `mix OBJ`
