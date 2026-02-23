---
name: scienceworld-mixture-creator
description: This skill chemically mixes the contents of a container using the 'mix' action. It should be triggered when all required ingredients (e.g., sodium chloride and water) are present inside a container and the agent needs to combine them to produce a new substance. The skill performs the mixing operation and outputs the resulting product, completing the synthesis step of the task.
---
# Skill: Chemical Mixture Creator

## Purpose
Execute the `mix` action on a container to chemically combine its contents into a new substance, as defined by the environment's chemistry system. This skill is the final step in a synthesis task after all required ingredients have been gathered into a single container.

## When to Use
- **Trigger Condition:** You have confirmed that a container in your inventory or the environment holds all the required chemical ingredients for a target substance.
- **Prerequisite:** You have already used `move` or `pour` actions to place the correct ingredients into the target container.
- **Verification:** You have used `examine <container>` and observed the required ingredients listed in its contents.

## Core Instruction
1.  Ensure the target container is in your inventory or accessible in the current room.
2.  Execute the action: `mix <container>`.
3.  The environment will process the chemistry and output the resulting substance.

## Important Notes
- This skill only performs the final `mix` action. Locating ingredients, finding containers, and combining them are separate exploration and manipulation tasks.
- The `mix` action is deterministic. If the correct ingredients are present, the reaction will always succeed.
- After mixing, you may need to `focus on <new_substance>` or `examine <container>` to confirm the task is complete.
- Do not attempt to `mix` a container that does not contain the precise ingredients listed in the recipe.

## Related Reference
For common chemical recipes (e.g., salt water = sodium chloride + water), see `references/common_recipes.md`.
