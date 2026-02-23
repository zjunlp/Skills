# Receptacle Suitability Guide

## Derived from Execution Trajectory
This guide summarizes the decision logic for selecting a storage receptacle, based on the observed agent behavior.

## Core Principles
1.  **Primary Storage:** Drawers are the primary candidate for storing utensils like knives, forks, and spoons.
2.  **Suitability Check Protocol:**
    a. **Go to** the candidate receptacle.
    b. If closed, **open** it.
    c. **Observe contents.**
    d. **Evaluate:**
        - **Empty:** Suitable for storage. Proceed with the `alfworld-object-storer` skill.
        - **Contains Similar Items:** May be suitable (e.g., a spoon in a drawer could indicate a utensil drawer). Use context to decide.
        - **Contains Unrelated Items:** Likely unsuitable. Close the receptacle and continue searching.
    e. If unsuitable, **close** the receptacle before leaving.

## Common Receptacle Types & Heuristics
- **Drawers (1-N):** Default search location for cutlery and small kitchen tools. Search sequentially.
- **Countertops:** Typically are **not** storage locations. They are surfaces for temporary placement.
- **Cabinets:** Potential for larger item storage. Not explored in the base trajectory but follow the same check protocol.
- **Sinkbasin:** A cleaning location, **not** a storage location.
- **Dining Table:** A surface for items in use, **not** for permanent storage.

## Flowchart for Decision Making
