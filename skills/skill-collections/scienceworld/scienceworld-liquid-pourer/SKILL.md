---
name: scienceworld-liquid-pourer
description: This skill transfers the contents of a source liquid container into a target container for mixing or preparation. It should be triggered when combining multiple substances, such as paints or chemicals, into a single vessel.
---
# Instructions for Liquid Pouring

This skill orchestrates the transfer of a liquid from a source container to a target container, typically as a preparatory step for mixing or chemical combination.

## Core Action
The primary action is `pour OBJ into OBJ`. Use this to transfer the contents.

## Execution Flow
1.  **Identify Containers:** Locate the source container (holding the liquid to be transferred) and the target container (the destination vessel).
2.  **Perform Transfer:** Execute the `pour` action with the correct object identifiers.
3.  **Verify (Optional):** If necessary, examine the target container to confirm the transfer was successful.

## Key Considerations
*   Ensure the target container is empty or can receive the new substance without adverse reaction (e.g., contamination).
*   The skill is often used in sequence with other skills, such as `mix`, to achieve a final compound.
*   Object identifiers (e.g., `wood cup (containing red paint)`) must be precise. Use `look around` and `examine` actions to confirm object states and contents if unsure.

## Example Usage
*   **Goal:** Create orange paint by mixing red and yellow.
*   **Procedure:**
    1.  `pour wood cup (containing red paint) into jug`
    2.  `pour wood cup (containing yellow paint) into jug`
    3.  `mix jug` (This is a separate mixing skill, not part of the pour operation).
