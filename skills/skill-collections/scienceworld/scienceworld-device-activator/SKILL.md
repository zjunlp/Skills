---
name: scienceworld-device-activator
description: This skill activates a device (e.g., blast furnace, stove) to initiate a process like heating. It should be triggered when the agent has placed materials in a device and needs to start the device's operation. The skill takes a device name as input and outputs a confirmation of activation, which is critical for progressing tasks that require energy or processing.
---
# Device Activation Skill

## Purpose
Activate a specific device to begin its operation (e.g., heating, processing). This is a critical step when materials have been placed into a device and require energy input to proceed with an experiment or task.

## When to Use
- You have confirmed that the target materials are properly placed inside the device.
- The device is in a deactivated or "off" state.
- The next step in your task requires the device to be operating (e.g., heating lead in a blast furnace).

## Core Instruction
Execute the `activate` action on the target device.

**Action Format:** `activate <DEVICE_NAME>`

**Example:** `activate blast furnace`

## Prerequisites
1. **Material Placement:** Ensure the target material(s) are inside the device. Use `look at <DEVICE_NAME>` to verify contents.
2. **Device State:** Confirm the device is deactivated. The observation will typically indicate "which is turned off" or "which is deactivated."
3. **Safety Check:** Ensure the device door is open if required for loading. Most devices in this environment are pre-opened.

## Post-Activation Steps
1. **Verification:** After activation, check the device state with `look at <DEVICE_NAME>` or `examine <DEVICE_NAME>` to confirm it's now "turned on" or "activated."
2. **Monitoring:** Use appropriate measurement tools (e.g., thermometer) on the materials inside the device to track progress.
3. **Deactivation:** Remember to deactivate the device when the process is complete using `deactivate <DEVICE_NAME>`.

## Common Devices & Context
- **Blast Furnace:** For high-temperature melting (e.g., metals like lead).
- **Stove/Oven:** For general heating tasks.
- **Other Devices:** Any device with an observable on/off state that requires activation to function.

For detailed device specifications and safety guidelines, see the reference documentation.
