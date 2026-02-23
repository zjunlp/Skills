# Action Execution Guide

## After Evaluation: What to Do Next

Once the threshold evaluation is complete, you must execute the appropriate action based on the result.

### 1. Identify the Target Action

From the original instruction, identify the two possible actions:
- **Action A**: What to do if the condition is `True` (value above/below threshold)
- **Action B**: What to do if the condition is `False` (value not above/below threshold)

**Example from trajectory:**
- If temperature > 50.0: `focus on green box`
- If temperature < 50.0: `focus on blue box`

### 2. Construct the Action Command

Use the standard action format from the environment:
- `focus on OBJ`
- `use OBJ`
- `activate OBJ`
- `move OBJ to OBJ`
- etc.

**Important:** Use the exact object names as they appear in the environment observations.

### 3. Execute Immediately

Do not insert intermediate steps between evaluation and action execution.

### 4. Verification (Optional)

After executing the action, you may want to:
- Confirm the action was successful (check observation)
- Ensure you're now on the correct branch of the task

## Common Action Patterns

| Condition Type | Typical Actions |
|----------------|-----------------|
| **Temperature** | `focus on [object]`, `activate/deactivate [heater/cooler]`, `move [sample] to [location]` |
| **Weight/Mass** | `use [container]`, `move [sample] to [scale]`, `pour [substance] into [vessel]` |
| **pH/Concentration** | `add [reagent]`, `mix [solution]`, `focus on [indicator]` |
| **Time/Duration** | `activate/deactivate [timer]`, `wait`, `focus on [clock]` |

## Error Recovery

If the evaluation leads to an impossible action:
1. **Re-check the measurement** - Did you extract the correct value?
2. **Re-parse the condition** - Did you identify the correct threshold and operator?
3. **Verify object existence** - Is the target object present in the current room?
4. **Consult trajectory** - Look for similar patterns in successful executions.

## Example: Complete Execution Flow

**Initial State:**
- Instruction: "If voltage > 12V, connect to red terminal. If voltage < 12V, connect to black terminal."
- Observation: "multimeter reads 14.5 volts"

**Skill Execution:**
1. Extract measurement: `14.5`
2. Parse condition: threshold=`12.0`, operator=`>`
3. Evaluate: `14.5 > 12.0` = `True`
4. Identify action: `connect to red terminal`
5. Construct command: `connect multimeter to red terminal`
6. Execute: `connect multimeter to red terminal`

**Result:** The correct branch is executed based on the measured value.
