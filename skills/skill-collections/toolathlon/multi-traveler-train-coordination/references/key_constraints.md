# Core Constraints to Enforce

1.  **Traveler Specifics**: Each traveler has defined origin and return stations.
2.  **Destination City Coordination**:
    *   All travelers must arrive at the **same station** within the destination city on the outbound leg.
    *   All travelers must depart from the **same station** within the destination city on the return leg.
    *   The arrival and departure stations in the destination city can be different.
3.  **Time Windows**:
    *   Outbound: Departure must be after a specified time (e.g., > 17:00).
    *   Return: Departure must be within a specified window (e.g., between 14:00 and 18:00).
4.  **Synchronization Tolerance**:
    *   Arrival times at the destination must be within ΔT (e.g., ≤ 30 minutes).
    *   Departure times from the destination must be within ΔT (e.g., ≤ 30 minutes).
5.  **Train Type**: Typically restricted to high-speed (G) or bullet (D) trains.
6.  **Output**: Find **one** valid combination per day. Return `null` for a day if no combination meets all constraints.
