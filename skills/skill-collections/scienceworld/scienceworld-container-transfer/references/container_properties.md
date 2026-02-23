# Reference: Common Container Properties

This document lists typical containers found in the ScienceWorld environment and their inferred properties to guide selection for the transfer skill.

## Container Types & Suitability

| Container          | Typical Material | Heat Resistance | Suggested Use Case                                  |
| :----------------- | :--------------- | :-------------- | :-------------------------------------------------- |
| **Metal Pot**      | Metal            | **High**        | Heating substances in a furnace or on a stove.      |
| **Ceramic Cup**    | Ceramic          | Moderate        | Holding hot liquids (not for direct flame).         |
| **Glass Jar/Cup**  | Glass            | Low-Variable    | Storage, mixing; may crack with rapid temp change.  |
| **Tin Cup**        | Metal (thin)     | Low             | General storage; not suitable for high-temperature heating. |
| **Wood Cup**       | Wood             | **Low**         | Cold storage only (e.g., in a fridge).              |
| **Bowl**           | Variable         | Low             | Holding solid items, mixing (check material).       |
| **Drawer**         | Wood/Metal       | N/A             | Storage of tools/items, not for active processing.  |

## Selection Heuristic
When choosing a destination container for a transfer, ask:
1.  **What is the next operation?** (e.g., heating → requires heat-resistant container).
2.  **Is the container empty?** Check via `look at <container>`.
3.  **Is the container accessible?** It must be in the current room and not inside another closed object.

## Example from Trajectory
*   **Operation:** Heat lead to melting point.
*   **Source:** `tin cup` (not suitable for blast furnace).
*   **Destination Selected:** `metal pot` (suitable for blast furnace).
*   **Command:** `move lead to metal pot`
