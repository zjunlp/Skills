# Common State Transitions in ScienceWorld

This document lists observable state changes for common substances and processes. Use these as reference when interpreting the output of `examine` commands.

## Heating/Melting Processes
| Substance | Initial State | Transition State | Final State (Melted) |
| :--- | :--- | :--- | :--- |
| Chocolate | `chocolate` | `softening chocolate` (optional) | `liquid chocolate` |
| Ice / Water | `ice` | `water` (melting) | `water` |
| Wax | `wax` | `soft wax` | `liquid wax` |
| Butter | `butter` | `softened butter` | `melted butter` |

## Chemical Reactions
| Process | Initial State | Transition Indicator | Final State |
| :--- | :--- | :--- | :--- |
| Combustion | `substance` + `flame` | `smoke`, `ashes` | `ashes` |
| Dissolution | `solid` in `liquid` | `dissolving solid` | `solution` |
| Precipitation | `solution` | `cloudy solution`, `precipitate forming` | `solution with precipitate` |

## Key Observation Patterns
*   **Adjective Change**: Look for new adjectives (`liquid`, `soft`, `cloudy`, `dissolved`).
*   **Noun Change**: The primary noun may change (`ice` → `water`).
*   **Compound Description**: The description may become a compound noun (`liquid chocolate`).

**Note**: The exact textual output is environment-dependent. Focus on detecting *any change* from the previous `examine` result.
