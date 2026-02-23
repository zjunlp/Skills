# Action Patterns for Planting

## Core Planting Action
The fundamental action for this skill is:
`move <SEED_SOURCE> to <CONTAINER>`

**Example:** `move banana seed in seed jar to flower pot 1`

## Resolving Ambiguity
The environment may generate ambiguous requests when multiple identical objects are present. The standard resolution pattern is:

1.  The agent issues a `move` command.
2.  The environment responds with a numbered list of possible interpretations.
3.  The agent selects the intended action by issuing the corresponding number (e.g., `0`).

**Trajectory Example:**
