# Guide: Testing Electrical Conductivity in ScienceWorld

This reference outlines a method for determining if an object is electrically conductive, as demonstrated in the skill trajectory. Use this logic **before** triggering the `scienceworld-object-classifier` skill.

## Required Components
A basic series circuit can be constructed to test conductivity. Common components found in the `workshop` include:
*   **Power Source:** Battery (has `anode` and `cathode` terminals).
*   **Conductors:** Wires (e.g., `blue wire`, `yellow wire`, `black wire`). Each wire has two terminals (`terminal 1`, `terminal 2`).
*   **Load/Indicator:** An `electric motor` or `light bulb` (has `anode` and `cathode` terminals). The motor turning on or the bulb lighting up indicates a complete circuit.
*   **Test Object:** The object in question (e.g., `metal pot`).

## Test Procedure
1.  **Construct Base Circuit:**
    *   Connect the battery's `anode` to one terminal of a wire.
    *   Connect that wire's other terminal to the `cathode` of the load (motor/bulb).
    *   Connect the battery's `cathode` to one terminal of a second wire.
    *   Connect the load's `anode` to one terminal of a third wire.
    *   **At this point, the circuit is open.** The second and third wires have free terminals.

2.  **Integrate Test Object:**
    *   Connect the free terminal of the wire from the battery's cathode (e.g., `yellow wire terminal 2`) to one terminal of the test object.
    *   Connect the free terminal of the wire from the load's anode (e.g., `black wire terminal 1`) to the other terminal of the test object.
    *   This places the test object in series within the circuit.

3.  **Observe and Classify:**
    *   **If Conductive:** The circuit is complete. The load (e.g., `electric motor`) will turn `on`.
    *   **If Non-Conductive:** The circuit remains open. The load will stay `off`.
    *   Use `look around` or `examine electric motor` to check its state (`on`/`off`).

## Classification Rule
*   **Conductive Object (`on` state):** Place in the `yellow box`.
*   **Non-Conductive Object (`off` state):** Place in the `purple box`.
