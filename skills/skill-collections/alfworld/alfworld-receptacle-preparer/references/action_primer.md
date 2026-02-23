# ALFWorld Action Reference

This document lists the core actions available to the agent, as defined in the trajectory. Use this for quick reference.

## Action List
1.  `go to {recep}` - Move to a specified receptacle.
2.  `take {obj} from {recep}` - Pick up an object from a receptacle.
3.  `put {obj} in/on {recep}` - Place an object into or onto a receptacle.
4.  `open {recep}` - Open a closed receptacle (e.g., drawer, fridge).
5.  `close {recep}` - Close an open receptacle.
6.  `toggle {obj} {recep}` - Activate a toggleable object (e.g., lightswitch, stove).
7.  `clean {obj} with {recep}` - Clean an object using a receptacle/tool.
8.  `heat {obj} with {recep}` - Heat an object using a receptacle/appliance.
9.  `cool {obj} with {recep}` - Cool an object using a receptacle/appliance.

## For Receptacle Preparation
The most critical actions for this skill are:
*   `go to {target_receptacle}`: Essential for step 1.
*   `open {target_receptacle}`: The primary preparatory action if the receptacle is found closed.
*   `toggle {obj} {recep}`: May be needed for certain electronic or latch mechanisms.

**Note:** The `{obj}` and `{recep}` placeholders must be replaced with the exact nouns observed in the environment (e.g., `garbagecan 1`, `sidetable 2`, `pen 3`).
