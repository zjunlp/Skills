# ALFWorld Action Specification
This document details the exact action grammar the agent must use.

## Available Actions
The agent can ONLY use actions from this list. `{obj}` and `{recep}` must be replaced with exact identifiers from observations.

1.  `go to {recep}`
2.  `take {obj} from {recep}`
3.  `put {obj} in/on {recep}`
4.  `open {recep}`
5.  `close {recep}`
6.  `toggle {obj} {recep}`
7.  `clean {obj} with {recep}`
8.  `heat {obj} with {recep}`
9.  `cool {obj} with {recep}`

## Formatting Rules
*   **Output Format:** Every agent response must be:
    