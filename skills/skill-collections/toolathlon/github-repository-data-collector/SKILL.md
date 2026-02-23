---
name: github-repository-data-collector
description: When the user needs to collect specific repository information from GitHub for analysis, reporting, or archival purposes. This skill fetches repository data from the GitHub API using repository IDs, extracts key metrics (name, owner, stars, forks, creation time, description, language, URL), handles missing/inaccessible repositories gracefully by skipping them, and saves the formatted data to a JSON file with specified structure. Triggers include GitHub repository data collection, milestone repository tracking, API data fetching from GitHub, JSON file creation with repository metrics.
---
# Instructions

## Objective
Collect specific GitHub repository information by ID from the GitHub API, extract defined metrics, and save the results to a structured JSON file. Skip any repositories that are inaccessible.

## Core Workflow

1.  **Parse the Request:** Identify the target repository IDs and the desired output file name (default: `github_info.json`). The user may specify these directly or imply them through context (e.g., "collect info for repos 1, 1000, 1000000").

2.  **Fetch Data:** For each repository ID, call the GitHub API endpoint: `GET https://api.github.com/repositories/{ID}`.
    *   Use the `fetch-fetch_json` tool.
    *   **Error Handling:** If the API returns an error (e.g., 404), log a note and skip this repository. Do not include it in the final output.

3.  **Transform Data:** For each successful API response, extract the following fields and map them to the specified JSON keys:
    *   `repo_name` -> `name`
    *   `owner` -> `owner.login`
    *   `star_count` -> `stargazers_count` (integer)
    *   `fork_count` -> `forks_count` (integer)
    *   `creation_time` -> `created_at` (ISO 8601 format, ensure it ends with 'Z')
    *   `description` -> `description` (can be `null`)
    *   `language` -> `language` (can be `null`)
    *   `repo_url` -> `html_url`

4.  **Build Output Structure:** Construct a JSON object where the top-level keys are the repository IDs (as strings). The value for each key is an object containing the 8 key-value pairs defined above.

5.  **Save File:** Write the final, formatted JSON object to the specified file path using `filesystem-write_file`. Use proper JSON formatting with indentation for readability.

6.  **Summarize:** Provide a concise summary to the user, confirming the file was saved and listing which repositories were successfully collected and which were skipped.

## Key Constraints & Notes
*   **Output Format:** The JSON must strictly follow the structure: `{ "<repo_id>": { "key_0": "value", ... } }`.
*   **Skipping Repos:** Only include data for repositories that return a successful API response. Do not create placeholder entries for failed requests.
*   **Data Types:** Ensure `star_count` and `fork_count` are integers. Preserve `null` values for `description` and `language` if the API provides them.
*   **Idempotency:** The skill can be run multiple times; it will overwrite the specified output file.

## Example
**User Request:** "Collect info for repos 1, 1000, and 1000000, save to `milestones.json`."
**Agent Action:** Fetches data for IDs 1, 1000, 1000000. If repo 1000 fails, the final `milestones.json` will only contain keys `"1"` and `"1000000"`.
