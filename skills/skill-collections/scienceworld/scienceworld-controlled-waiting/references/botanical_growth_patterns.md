# Botanical Growth & Pollination Patterns

## Observed Timeline (Derived from Trajectory)
The following pattern was observed for banana plant growth in the provided simulation environment. Use this as a heuristic for similar tasks.

1.  **Planting to Reproduction:** ~30 simulation steps (3 x `wait` actions).
    *   Action Sequence: Plant seed -> `wait` (x3) -> Plant reaches "reproducing stage" with flowers.

2.  **Flowering to Fruit Set:** Variable, requires pollination trigger.
    *   Observation: Flowers persisted through multiple `wait` cycles without fruit.
    *   **Trigger Identified:** Closing environment doors (containing bees) acted as a pollination catalyst.
    *   Post-trigger, fruit (bananas) appeared within ~20-30 steps (2-3 x `wait` actions).

3.  **Key Insight:** Some growth stages are purely time-based, while others (like fruit set) require specific environmental conditions *in addition* to time.

## General Guidelines for Controlled Waiting
*   **Baseline Wait:** Start with 2-3 `wait` commands for any biological growth stage transition.
*   **Check Point:** Always perform a `look around` or `examine` after a baseline wait to assess progress.
*   **Stalled Progress:** If no change occurs after two baseline cycles, investigate environmental prerequisites (e.g., pollinators, water, light, closed containers).
*   **Pollination Note:** The presence of bees (`adult bee`) and an open `bee hive` suggests a pollination mechanic. Containing them (`close door`) may be necessary to initiate the process.
