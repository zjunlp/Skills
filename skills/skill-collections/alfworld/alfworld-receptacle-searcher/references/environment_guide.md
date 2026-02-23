# ALFWorld Environment Guide for Receptacle Searching

## Overview
This document provides context about the ALFWorld environment to aid in receptacle searching. It is intended to be loaded into the agent's context when needed.

## Common Receptacle Types
1. **Cabinets (cabinet 1-20)**: Often closed; may contain items. The garbage can is typically not inside a cabinet.
2. **Countertops (countertop 1, 2)**: Usually open surfaces with items placed on them.
3. **Tables (diningtable 1, 2)**: Similar to countertops.
4. **Appliances**: Fridge, microwave, stoveburner, toaster, coffeemachine.
5. **Containers**: Drawer 1-6, sinkbasin, garbagecan.

## Search Heuristics
- **Garbage Can**: Often a standalone receptacle (`garbagecan 1`). It is usually not inside another receptacle. In the trajectory, it was found by going directly to it after checking many cabinets.
- **Cabinets**: If the target is a cabinet, you may need to open it first (`open cabinet X`).
- **Efficiency**: When searching for a receptacle that is likely to be in the open (e.g., garbage can, countertop), prioritize checking open surfaces first before enclosed spaces.

## Action Set (from Trajectory)
The available actions relevant to searching:
- `go to {recep}`: Navigate to a receptacle.
- `open {recep}`: Open a closed receptacle (e.g., cabinet, fridge).
- `close {recep}`: Close an open receptacle.

## Observation Patterns
- "On the {recep}, you see ...": The receptacle is open and has items.
- "The {recep} is closed.": The receptacle is closed; you may need to open it to see inside.
- "Nothing happened.": The action was invalid (e.g., trying to go to a non-existent receptacle).

## Example Trajectory Snippet
