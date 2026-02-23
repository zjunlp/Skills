# Alfworld Action Guide & Conventions

## Core Action Set
The agent can perform the following atomic actions. `{obj}` and `{recep}` must be replaced with exact names from observations.

1.  `go to {recep}` - Navigate to a receptacle.
2.  `take {obj} from {recep}` - Pick up an object from a receptacle.
3.  `put {obj} in/on {recep}` - Place a held object onto/in a receptacle.
4.  `open {recep}` - Open a closed receptacle (e.g., drawer, fridge).
5.  `close {recep}` - Close an open receptacle.
6.  `toggle {obj} {recep}` - Interact with a toggleable object (e.g., switch on a lamp).
7.  `clean {obj} with {recep}` - Clean an object using a receptacle/tool.
8.  `heat {obj} with {recep}` - Heat an object using a receptacle/appliance.
9.  `cool {obj} with {recep}` - Cool an object using a receptacle/appliance.

## Naming Conventions
*   Entities are always named with a **base type** and an **instance number** (e.g., `laptop 1`, `sidetable 2`).
*   The simulator uses the exact string. `armchair 1` is different from `armchair1`.
*   Observations list entities with an indefinite article "a", even before vowels (e.g., "a armchair 1"). Use the entity name *without* the article in actions.

## Observation Semantics
*   `On the {recep}, you see...` - Lists objects currently on that receptacle.
*   `You see nothing.` - The receptacle is empty.
*   `Nothing happened.` - The last action was **invalid** (e.g., object not found, precondition not met). This is critical feedback for replanning.
*   `You pick up the {obj} from the {recep}.` - Confirmation of a successful `take` action.
*   `You open the {recep}.` - Confirmation of a successful `open` action.

## Strategic Implications for Scanning
1.  **Receptacle Primacy:** The `go to` action only accepts receptacle targets. Therefore, identifying all receptacles is the first step for navigation.
2.  **Object Discovery:** Objects are only revealed when you are at (`go to`) a receptacle and receive the "On the..." observation. The initial scan provides the list of locations to check.
3.  **Containment Hierarchy:** Some receptacles (like `drawer`) may be closed initially, requiring an `open` action before their contents are known. The scanner should note these for potential future exploration.
