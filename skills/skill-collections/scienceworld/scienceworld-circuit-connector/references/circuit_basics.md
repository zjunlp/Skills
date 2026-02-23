# Circuit Building Basics

## Common Components & Terminals
| Component | Typical Terminals | Notes |
| :--- | :--- | :--- |
| **Battery** | `anode` (positive, +), `cathode` (negative, -) | Power source. |
| **Wire** | `terminal 1`, `terminal 2` | Conducts current. Color is cosmetic. |
| **Light Bulb** | `anode`, `cathode` | Lights up when current flows. |
| **Buzzer** | `anode`, `cathode` | Sounds when current flows. |
| **Switch** | `terminal 1`, `terminal 2` | Must be `activate`d to close the circuit. |
| **Unknown Substance** | `terminal 1`, `terminal 2` | Device Under Test (DUT). Conductivity is unknown. |

## Standard Conductivity Test Circuit
This is the circuit built in the provided trajectory to test an unknown substance (M).
1.  Connect `battery anode` to `black wire terminal 1`.
2.  Connect `battery cathode` to `green wire terminal 1`.
3.  Connect `black wire terminal 2` to `buzzer cathode`.
4.  Connect `orange wire terminal 2` to `buzzer anode`.
5.  Connect `unknown substance M terminal 1` to `green wire terminal 2`.
6.  Connect `unknown substance M terminal 2` to `orange wire terminal 1`.

**Circuit Path:** Battery (+) -> Black Wire -> Buzzer -> Orange Wire -> Substance M -> Green Wire -> Battery (-).

**Interpretation:** If the substance is conductive, the circuit is complete, and the buzzer will activate (sound). If it is non-conductive, the circuit remains open, and the buzzer stays off.

## Troubleshooting
*   **No Activation:** Ensure all connections are made correctly and the power source (battery) is functional. Verify the switch (if present) is activated.
*   **Incorrect Terminal:** Double-check terminal names by using the `examine OBJ` action.
*   **Disconnection:** Use `disconnect OBJ` to remove a specific connection from a component.
