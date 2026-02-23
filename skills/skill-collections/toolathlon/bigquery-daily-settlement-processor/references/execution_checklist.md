# Execution Checklist & Common Pitfalls

Use this list to ensure robust execution of the settlement skill.

## Pre-flight Checks
- [ ] **Dataset Confirmation:** Have you listed datasets and confirmed the target one exists?
- [ ] **Source Data Inspection:** Have you run `SELECT ... LIMIT` on the fully-qualified source table (`dataset.daily_scores_stream`)?
- [ ] **Date Confirmation:** Have you explicitly determined the target processing date (e.g., via `CURRENT_DATE()` or user input)?
- [ ] **Schema Validation:** Have you checked the schema of the `player_historical_stats` table to confirm column names and order?

## Critical Pitfalls to Avoid
1.  **Unqualified Table Names:** Never run a query on `daily_scores_stream` alone. Always use `dataset.daily_scores_stream`.
2.  **Incorrect Date Format:** Ensure the date in the `WHERE` clause uses the ISO format `'YYYY-MM-DD'`. The derived date for the table name should be `'YYYYMMDD'`.
3.  **Schema Mismatch:** The `INSERT` query column list must match the exact schema of `player_historical_stats`. The script generates the correct query.
4.  **Assuming Today's Data:** Do not assume the latest data is for "today". Always verify the `latest_data_date` from the source stream.

## Verification Steps (Post-Execution)
- [ ] **Leaderboard Created:** Run `SELECT COUNT(*) FROM dataset.leaderboard_YYYYMMDD`. Result should be 100.
- [ ] **Stats Inserted:** Run `SELECT COUNT(*) FROM dataset.player_historical_stats WHERE date = 'YYYY-MM-DD'`. Result should be > 0.
- [ ] **Sample Output:** Provide the user with the top 5 rows from the new leaderboard as proof of successful execution.
