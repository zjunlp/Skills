# ALFWorld Action Primer

## Core Action Set
The agent can perform the following atomic actions. The `{obj}` and `{recep}` placeholders must be replaced with specific object and receptacle names observed in the environment.

1.  `go to {recep}` - Move to a specified receptacle (e.g., `go to desk 1`).
2.  `take {obj} from {recep}` - Pick up an object from a receptacle.
3.  `put {obj} in/on {recep}` - Place an object into or onto a receptacle.
4.  `open {recep}` - Open a closed receptacle (e.g., a drawer).
5.  `close {recep}` - Close an open receptacle.
6.  `toggle {obj} {recep}` - Toggle the state of an object (e.g., turn on a lamp).
7.  `clean {obj} with {recep}` - Clean an object using a receptacle/tool.
8.  `heat {obj} with {recep}` - Heat an object using a receptacle/tool.
9.  `cool {obj} with {recep}` - Cool an object using a receptacle/tool.

## Navigation-Specific Guidance
- The `go to` action is the primary movement command.
- Receptacles are fixed locations (`desk 1`, `shelf 1`) or containers (`drawer 2`).
- Before interacting with an object on a receptacle, you must first `go to` that receptacle.
- If an action fails with "Nothing happened", the target may be invalid, inaccessible, or require a prerequisite action (e.g., `open` before `take` from a closed drawer).

## Common Receptacle Types & Associations
| Receptacle    | Typical Contents                              |
|---------------|-----------------------------------------------|
| `desk 1`      | desklamp, book, laptop, pen                   |
| `bed 1`       | pillow, book, cellphone                       |
| `sidetable 1` | alarm clock, CD, keychain                     |
| `shelf 1`     | pen, book                                     |
| `drawer 1/2`  | CD, credit card, small items                  |
| `garbagecan 1`| (often empty)                                 |

## Response Format
The agent must always output its reasoning and action in the following strict format:
