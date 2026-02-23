# Skill Application Example

The following trajectory snippet from the `heat some potato` task demonstrates the `alfworld-appliance-preparer` skill in action. The core skill steps are highlighted.

**Goal:** Heat some potato and put it in diningtable.

1.  **[Prerequisite]** Agent acquires a potato.
2.  **[Skill Trigger]** Agent needs to use the `microwave 1` to heat the potato.
3.  **[Skill Execution - Prepare Appliance]**
    *   `Thought:` Now that I have the potato, I need to find an appliance to heat it. The microwave is a suitable option for heating food quickly.
    *   `Action: go to microwave 1`
    *   `Observation:` The microwave 1 is closed.
    *   `Thought:` Before I can heat the potato, I need to open the microwave to place the potato inside. *(Decision to open the appliance)*
    *   `Action: open microwave 1` *(Core preparation action)*
    *   `Observation:` You open the microwave 1. The microwave 1 is open. In it, you see nothing.
    *   **Skill Outcome:** The microwave is now successfully prepared (open) and ready to receive an item for heating.
4.  **[Post-Skill Action]** The agent proceeds to `heat potato 3 with microwave 1`.

**Key Takeaway:** The skill encapsulates the sequence of navigating to the appliance, checking its state (`is closed`), and taking the correct action (`open`) to make it ready for the next operation (`heat`).
