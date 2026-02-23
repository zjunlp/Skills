---
name: controlled-waiting
description: Executes timed waiting to allow processes like plant growth or pollination to progress. Use this skill when you need to advance time for biological or mechanical processes to reach their next stages. This enables progression in tasks that require temporal development rather than immediate actions.
---
# Skill: Controlled Waiting

## Purpose
This skill orchestrates a strategic waiting pattern to allow time-dependent processes (e.g., plant growth, chemical reactions, mechanical cycles) to complete. It is triggered when the agent's primary task is blocked, pending the natural progression of a state.

## Core Logic
The skill follows a **Monitor-Wait-Check** cycle:
1.  **Assess State:** Confirm the process requiring time is active (e.g., seeds are planted, device is activated).
2.  **Execute Wait:** Use the `wait` action (for long periods) or `wait1` (for single steps) to advance the simulation.
3.  **Verify Progress:** After waiting, check the environment (`look around` or `examine`) to see if the target state has been reached.
4.  **Repeat or Exit:** If the target state is not yet achieved, loop back to step 2. If achieved, exit the skill and resume the main task.

## Key Parameters & Decisions
*   **Wait Duration:** Use `wait` (10 steps) for significant biological/mechanical stages. Use `wait1` for fine-grained control or to observe rapid changes.
*   **Monitoring Frequency:** The interval between checks. Derived from the trajectory: after 2-3 `wait` actions, a `look around` is performed. Adjust based on the estimated time of the process.
*   **Exit Condition:** Clearly defined by the main task goal (e.g., "banana is present", "device status is 'ready'").

## When to Use This Skill
*   After initiating a process that has a known or unknown delay.
*   When the environment state is stable and no other preparatory actions are possible.
*   When prompted by task context (e.g., "give them time to grow", "wait for the reaction to complete").

## When NOT to Use This Skill
*   When you can perform other productive actions in parallel.
*   When the process is instantaneous or requires a specific trigger (e.g., pressing a button).
*   If waiting would cause a negative outcome (e.g., a timer expires, an object decays).

## Integration with Main Task
This skill is a **subroutine**. The main task should:
1.  Set up the necessary conditions for the time-based process.
2.  Invoke this skill.
3.  Upon skill completion, verify the outcome and proceed with the next task step (e.g., harvest the grown banana).

Refer to `references/botanical_growth_patterns.md` for common plant growth stage timings.
