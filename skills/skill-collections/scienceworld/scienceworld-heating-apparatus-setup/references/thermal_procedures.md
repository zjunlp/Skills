# Thermal Procedure Reference

## Overview
This document provides supplementary context for heating-related experiments, detailing common substances, their properties, and best practices.

## Common Substance Melting/Boiling Points
*   **Chocolate:** Melting point typically ranges between 30°C and 32°C (86-90°F) for cocoa butter. Compound chocolate may have different properties.
*   **Water:** Boiling point is 100°C at standard atmospheric pressure.
*   **Sodium Chloride (Salt):** Melting point is 801°C.

## Heating Device Profiles
| Device    | Typical Use Case          | Activation Command | Notes |
|-----------|---------------------------|--------------------|-------|
| Stove     | Heating containers directly | `activate stove`   | Used for pots, pans. Provides direct heat. |
| Oven      | Enclosed, ambient heating | `activate oven`    | Used for baking, roasting. Heat surrounds container. |
| Hot Plate | Precise heating of labware| `activate hot plate`| Common in laboratory settings. |

## Safety & Best Practices
1.  **Container Selection:** Always use a container suitable for the heating device (e.g., metal pot for stove, glass beaker for hot plate).
2.  **Never Heat Closed Containers:** Ensure containers are open or vented to prevent pressure buildup.
3.  **Monitor Continuously:** Heating can cause rapid phase changes. Use the `monitor_temperature.py` script for consistent observation.
4.  **Deactivate After Use:** Once the experiment is complete, remember to `deactivate` the heating device.

## Troubleshooting
*   **Substance Not Heating:** Verify the container is correctly positioned *on* the heating element (use `look at <DEVICE>`).
*   **Temperature Not Rising:** Confirm the heating device is activated (`activate <DEVICE>`).
*   **Incorrect Phase Change:** Consult the melting/boiling point table above to set appropriate expectations.
