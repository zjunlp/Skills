# Appliance Interaction Guide

This guide details the prerequisites and common issues for using appliances to modify object states.

## Core Modifier Actions
The following actions are directly supported by the skill:
- `cool {obj} with {appliance}`
- `heat {obj} with {appliance}`
- `clean {obj} with {appliance}`

## Appliance-Specific Protocols

### Refrigerator (`fridge`)
*   **Purpose:** Cooling objects.
*   **Typical Prerequisites:**
    *   The fridge does **not** need to be open to perform the `cool` action externally.
    *   You must be holding the object.
    *   You must be at the fridge's location (`go to fridge 1`).
*   **Common Error:** Attempting `cool` from a distance or without the object.

### Microwave (`microwave`)
*   **Purpose:** Heating objects.
*   **Typical Prerequisites:**
    *   The microwave usually must be **open** to place an object inside *before* heating.
    *   The `heat` action may be performed after the object is placed inside (`put {obj} in/on microwave 1`).
*   **Note:** The trajectory shows `heat` as the modifier action. Placing an object inside is a separate step (`put`).

### Sink (`sinkbasin`)
*   **Purpose:** Cleaning objects.
*   **Typical Prerequisites:**
    *   You must be at the sink.
    *   You must be holding the object.
*   **Common Error:** Attempting to clean an object that is not dirty or using the wrong receptacle.

## Troubleshooting "Nothing happened"
If the environment returns "Nothing happened" after your modifier action, check this list:

1.  **Object in Hand?** Verify you have successfully taken the object (`take {obj} from {recep}`).
2.  **Correct Location?** Verify you are at the appliance (`go to {appliance}`).
3.  **Appliance State?** For microwaves/ovens, ensure they are open if required. For stoves, ensure the burner is toggled on.
4.  **Valid Combination?** Ensure the appliance can perform the requested state change (e.g., you cannot `heat` with a fridge).
5.  **Action Syntax?** Double-check the action format: `{state} {obj} with {appliance}`. Spelling and spacing matter.
