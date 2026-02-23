---
name: alfworld-device-operator
description: This skill operates a device or appliance (like a desklamp) to interact with another object. It should be triggered when the task requires using a tool on a target item (e.g., 'look at laptop under the desklamp'). The skill assumes the target object and the operating device are co-located, and it executes the appropriate use action (e.g., toggle, heat, clean).
---
# Instructions

## 1. Skill Trigger
Activate this skill when the task goal explicitly requires **using a device or appliance** to interact with a target object. Common indicators include phrases like:
- "look at [object] under the [device]"
- "heat [object] with [device]"
- "cool [object] with [device]"
- "clean [object] with [device]"

## 2. Core Execution Flow
Follow this sequence when the skill is triggered:

### Phase 1: Locate the Device
1.  **Identify the device** from the task description (e.g., `desklamp`, `microwave`, `fridge`).
2.  **Search common receptacles** where such a device is typically found (e.g., desks, sidetables, countertops).
3.  Use the `go to {recep}` action to navigate to and inspect these locations until the device is found.
4.  **Note the device's exact name** (e.g., `desklamp 1`).

### Phase 2: Locate the Target Object
1.  **Identify the target object** from the task description (e.g., `laptop`, `mug`, `plate`).
2.  **Search the environment** for this object. It may not be near the device initially.
3.  Use the `go to {recep}` and visual inspection to find the object.
4.  Once found, use `take {obj} from {recep}` to pick it up.

### Phase 3: Co-locate Object and Device
1.  **Navigate** to the receptacle where the target device is located using `go to {recep}`.
2.  Ensure you are in the same location as the device before proceeding.

### Phase 4: Operate the Device
1.  Execute the final **use action**. The specific action is determined by the device-object pair:
    *   For a `desklamp` and a viewable object (like a `laptop`, `book`), use: `toggle {device} {recep}`.
    *   For a `microwave` and a heatable object, use: `heat {obj} with {device}`.
    *   For a `fridge` and a coolable object, use: `cool {obj} with {device}`.
    *   For a `faucet`/`spraybottle` and a cleanable object, use: `clean {obj} with {device}`.
2.  **Action Format:** The action must use the exact object and device names discovered during Phases 1 & 2.

## 3. Key Assumptions & Rules
*   **Co-location Required:** The skill assumes the final action requires the target object and the operating device to be in the same location (on the same receptacle).
*   **Device First:** Prioritize finding the device before extensively searching for the target object, as the device's location is often a fixed landmark.
*   **Invalid Actions:** If the environment responds with "Nothing happened," re-evaluate your object/device names and your location. Ensure you are using the correct action verb for the device type.
