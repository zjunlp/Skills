# Reference: Phase Change Observation Protocol

## Overview
Cooling a substance to observe a phase change (e.g., liquid to solid) is a multi-step process. The `scienceworld-substance-cooler` skill is **Step 1**.

## Full Workflow
1.  **Initialization (This Skill):** Move the substance into a cooling appliance.
2.  **Monitoring:** Periodically use a thermometer on the substance to track its temperature.
    *   Command: `use thermometer on <SUBSTANCE_NAME>`
    *   Frequency: Varies. Monitor until the target temperature range is approached.
3.  **Observation:** Continuously `examine <SUBSTANCE_NAME>` to detect visual/physical state changes indicating a phase transition.
4.  **Decision:** Based on the observed melting/freezing point and task criteria, take the next action (e.g., "focus on yellow box").

## Key Concepts from Trajectory
*   **Melting/Freezing Point:** The temperature at which a substance changes state. The task may involve determining if this point is above or below a specific value (e.g., -10.0°C).
*   **Thermal Equilibrium:** It takes time for the substance to cool to the appliance's temperature. Repeated measurements are necessary.
*   **Appliance Limitations:** A standard kitchen freezer may not reach extremely low temperatures. For very low melting points, an "ultra low temperature freezer" might be required.

## Common Cooling Appliances
*   `freezer`: Standard kitchen appliance. Cools to temperatures below 0°C.
*   `ultra low temperature freezer`: Found in labs/workshops. Cools to much lower temperatures (e.g., -50°C or below).

## Error Handling
*   **"You can't do that.":** Ensure the appliance door is open. Use `look at <APPLIANCE>` to check.
*   **No Temperature Change:** The appliance might be off or broken. Try `activate <APPLIANCE>` if applicable.
*   **Substance Not Found:** Confirm the correct container description from the latest `look around` observation.
