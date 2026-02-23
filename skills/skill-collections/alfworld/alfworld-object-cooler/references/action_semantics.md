# Action Semantics for 'cool'

## Definition
The `cool {obj} with {recep}` action reduces the temperature of a held object (`{obj}`) by interacting with a cooling appliance (`{recep}`).

## Semantics from Trajectory
*   **Agent State:** The agent must be holding `{obj}`.
*   **Location:** The agent must be at the location of `{recep}`. The trajectory shows the agent at `fridge 1` before executing `cool pot 1 with fridge 1`.
*   **Receptacle State:** The receptacle (e.g., `fridge 1`) does not necessarily need to be open. The trajectory executed the action successfully while the fridge was closed.
*   **Object State:** The object is assumed to be in a state where cooling is a valid operation (e.g., a hot pot).

## Expected Outcomes
1.  **Success:** "You cool the {obj} using the {recep}." This indicates the object's state has changed.
2.  **Failure (Invalid Action):** "Nothing happened." This indicates a prerequisite was not met (wrong object, wrong location, invalid receptacle).

## Common Valid Receptacles
*   `fridge 1`
*   `freezer 1`
*   (Other environment-specific cooling appliances)

## Related Actions
*   `heat {obj} with {recep}`: The inverse operation.
*   `take {obj} from {recep}`: To acquire the object.
*   `go to {recep}`: To navigate to the cooling appliance.
*   `put {obj} in/on {recep}`: Likely the next step after cooling.
