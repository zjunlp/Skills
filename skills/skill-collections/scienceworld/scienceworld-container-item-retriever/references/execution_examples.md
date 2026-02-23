# Execution Examples & Edge Cases

## Primary Example (From Trajectory)
**Scenario:** Need an avocado seed from a seed jar in the greenhouse.
1.  **Observation Context:** `a seed jar (containing a avocado seed, a avocado seed, ...)`
2.  **Skill Execution:** `pick up seed jar`
3.  **Success Output:** `You move the seed jar to the inventory.`

**Note:** In the provided trajectory, the entire container was picked up. The skill principle remains the same for picking up a specific item *from* the container (e.g., `pick up avocado seed`), assuming the container is open.

## Alternative: Picking a Specific Item from Container
**Scenario:** Need one "adult bee" from an open bee hive.
1.  **Observation Context:** `a bee hive. The bee hive door is open. In the bee hive is: a adult bee, a adult bee, ...`
2.  **Skill Execution:** `pick up adult bee`
3.  **Expected Output:** `You move the adult bee to the inventory.`

## Edge Cases & Troubleshooting

### Case 1: Container is Closed
**Symptom:** `pick up` action fails or is not possible.
**Solution:** You must open the container first using `open <CONTAINER>` before this skill can be applied.

### Case 2: Item Not Found
**Symptom:** Action results in an error or unexpected observation.
**Pre-action Check:** Always verify the item is listed in the current room observation inside the specified container. Use `look around` or `look at <CONTAINER>` if unsure.

### Case 3: Ambiguous Object Names
**Symptom:** System prompts for clarification with a numbered list (as seen with `move` in the trajectory).
**Guidance:** The `pick up` action in the provided environment typically does not trigger this ambiguity for items in open containers. If it does, select the first option (`0`) or the option that correctly identifies the item in the target container.

### Case 4: Item is a Substance
**Symptom:** Item is described as "a substance called X" (e.g., water, air).
**Guidance:** Substances often cannot be picked up directly. They may require a container (e.g., `pour` action). This skill is for discrete, tangible objects.
