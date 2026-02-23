# Table Schema Reference

This document details the expected schemas for tables involved in the daily settlement process, as observed in the execution trajectory.

## Source Table: `daily_scores_stream`
Stores raw, streaming game score events.
| Field | Type | Description | Notes |
| :--- | :--- | :--- | :--- |
| `timestamp` | TIMESTAMP | UTC timestamp of the game event. | Used to filter by date. |
| `scores` | RECORD (STRUCT) | A nested structure containing score components. | |
| `scores.task_score` | INTEGER | Score earned from tasks/completions. | |
| `scores.online_score` | INTEGER | Score earned from online/engagement metrics. | |
| `player_region` | STRING | Geographic region of the player (e.g., 'ASIA', 'US', 'EU', 'CN'). | |
| `game_id` | STRING | Unique identifier for the game session. | |
| `player_id` | STRING | Unique identifier for the player. | Primary key for aggregation. |

## Target Table: `player_historical_stats`
Master table storing aggregated daily statistics per player.
| Field | Type | Description | Notes |
| :--- | :--- | :--- | :--- |
| `player_id` | STRING | Unique identifier for the player. | Part of composite primary key with `date`. |
| `date` | DATE | The date the statistics represent. | Part of composite primary key with `player_id`. |
| `total_score` | INTEGER | Sum of `task_score` and `online_score` for the player on this date. | Derived field. |
| `game_count` | INTEGER | Number of distinct game sessions the player participated in on this date. | |
| `total_task_score` | INTEGER | Sum of all `task_score` values for the player on this date. | |
| `total_online_score` | INTEGER | Sum of all `online_score` values for the player on this date. | |
| `player_region` | STRING | The player's region. Assumed constant per day; use `MAX()` in aggregation. | |

## Output Table: `leaderboard_YYYYMMDD`
Daily snapshot of the top 100 players. Table name is dynamically generated (e.g., `leaderboard_20251126`).
| Field | Type | Description | Notes |
| :--- | :--- | :--- | :--- |
| `player_id` | STRING | Unique identifier for the player. | |
| `total_score` | INTEGER | The player's aggregated total score for the day. | |
| `rank` | INT64 | The player's rank for the day (1 is highest). | Calculated using `ROW_NUMBER()`. |
