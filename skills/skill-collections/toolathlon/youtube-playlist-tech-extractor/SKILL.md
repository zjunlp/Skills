---
name: youtube-playlist-tech-extractor
description: When the user requests analysis of a YouTube playlist to identify technical content, extract specific technologies, and locate their corresponding GitHub repositories. This skill enables browsing YouTube playlists, filtering for technical/machine learning videos, identifying mentioned technologies, searching for their GitHub repositories, and compiling structured documentation with project details, repository URLs, and main functions. Triggers include YouTube playlist URLs, requests to find technical/ML videos, needs to locate GitHub repositories for technologies, and requirements to create documentation about tech projects.
---
# Skill: YouTube Playlist Tech Extractor

## Core Workflow
1.  **Navigate & Parse Playlist:** Use the browser to load the provided YouTube playlist URL. Navigate through all spans to capture the complete list of video titles and metadata.
2.  **Filter & Identify Technologies:** Analyze the video titles to identify those related to technical, machine learning, or AI coding topics. Extract the specific technology names mentioned (e.g., "Claude Code", "FlashAttention", "Qwen").
3.  **Search for Repositories:** For each identified technology, search for its official or primary GitHub repository. Use the browser to navigate to likely repository URLs (e.g., `github.com/<org>/<project-name>`) to verify their existence and gather details.
4.  **Compile Documentation:** Create a structured markdown report. For each technology, document:
    *   The technology/project name.
    *   The official GitHub repository URL.
    *   A description of its main functions and capabilities, synthesized from the repository description and your knowledge.
    *   A list of related videos from the playlist.
5.  **Output:** Write the final compiled report to a specified file in the workspace (e.g., `ml_tech.md`).

## Key Instructions & Heuristics
*   **Playlist Navigation:** YouTube playlists are often paginated or loaded in "spans". Use the `browser_snapshot_navigate_to_next_span` tool iteratively until you have viewed all video entries.
*   **Technology Recognition:** Look for keywords in video titles: AI model names (Claude, Gemini, Qwen, Mistral), project names (OpenHands, FlashAttention), tools (GitHub Copilot), and technical domains (machine learning, coding, AI agents).
*   **Repository Discovery:** Construct repository URLs based on common patterns: `github.com/<organization>/<project-name>`. The organization is often the company or research lab behind the technology (e.g., `anthropics`, `Dao-AILab`, `QwenLM`). Verify each URL by navigating to it. If a 404 is encountered, the technology may be proprietary (e.g., Apple Intelligence) or hosted elsewhere; note this in the report.
*   **Report Structure:** Follow the template demonstrated in the trajectory. Include a summary table for quick reference. Clearly note when a technology is proprietary and lacks a public repository.

## Tools Required
*   `playwright_with_chunk-browser_navigate`: To load the initial playlist and repository pages.
*   `playwright_with_chunk-browser_snapshot_navigate_to_next_span`: To scroll through the entire playlist.
*   `playwright_with_chunk-browser_tab_new` / `playwright_with_chunk-browser_tab_select`: To efficiently open and switch between multiple GitHub repository pages.
*   `filesystem-write_file`: To create the final output markdown file.
*   `filesystem-read_file`: (Optional) To verify the written output.

## Error Handling & Edge Cases
*   If the playlist is private or requires login, inform the user.
*   If a technology name is ambiguous, search for the most likely repository and use the repository's own description to confirm.
*   If a repository search fails, note "Repository not found (likely proprietary/closed source)" in the report.
*   The final output file path should be confirmed with the user or derived from their request (e.g., `ml_tech.md` in the workspace).
