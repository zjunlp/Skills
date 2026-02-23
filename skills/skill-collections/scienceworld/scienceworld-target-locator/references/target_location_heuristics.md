# Target Location Heuristics for ScienceWorld

This reference contains common-sense mappings between object types and their most likely locations in the ScienceWorld environment. Use this to inform the `teleport` decision.

## Room List
kitchen, foundry, workshop, bathroom, outside, living room, bedroom, greenhouse, art studio, hallway

## Object Type -> Probable Room(s)

*   **animal / egg / living creature:** `outside`, `greenhouse`, `bedroom` (if pet)
*   **tool / device / mechanical part:** `workshop`, `foundry`, `kitchen`
*   **container (box, chest, bottle):** `workshop`, `kitchen`, `living room`, `bedroom`
*   **chemical / substance / liquid:** `bathroom`, `kitchen`, `workshop`, `greenhouse`
*   **food / ingredient:** `kitchen`, `greenhouse`
*   **art supply / canvas / paint:** `art studio`
*   **furniture / decorative item:** `living room`, `bedroom`, `hallway`
*   **plant / seed / soil:** `greenhouse`, `outside`
*   **book / note / paper:** `bedroom`, `living room`, `art studio`, `workshop`
*   **metal / ore / raw material:** `foundry`, `workshop`, `outside`

## Search Priority Notes
1.  `outside` and `greenhouse` are high-probability for biological targets.
2.  `workshop` and `foundry` are high-probability for mechanical/industrial targets.
3.  If a high-probability room is not directly accessible via a visible door, use `teleport`.
4.  If the target type is ambiguous, default to the first room in its list.
