---
name: youtube-playlist-tracklist-extractor
description: Extracts song lists or track information from YouTube videos or playlists, particularly for music compilations, trending playlists, or music mixes. Handles searching, navigating, extracting tracklists from descriptions, and formatting the data.
---
# Skill: YouTube Playlist Tracklist Extractor

## Primary Objective
Extract a structured list of songs from a specified YouTube video or playlist and write it to a file, following a provided format template.

## Core Workflow
1.  **Parse Request & Load Format:** Understand the user's request for a specific YouTube video/playlist. Immediately read the specified format file (e.g., `format.md`) to understand the required output structure.
2.  **Search & Identify Target:** Perform a web search using the exact or descriptive title provided by the user to find the correct YouTube URL.
3.  **Access Video & Extract Data:** Navigate to the identified YouTube video URL. Locate and expand the video description to reveal the full tracklist.
4.  **Parse & Structure Data:** Extract all song entries from the description. Clean and format the data (e.g., standardizing song numbers, artists, and titles).
5.  **Generate Output:** Compile the extracted song list into the structure defined by the format template and write it to the specified output file.
6.  **Verify & Conclude:** Read back the created file to confirm successful creation and content accuracy, then notify the user.

## Key Techniques & Handling
- **Description Expansion:** YouTube descriptions are often truncated. Look for and click buttons like "`...more`" or "`Show more`" to reveal the full text containing the tracklist.
- **Robust Parsing:** Tracklists in descriptions can have varying formats (e.g., "1. Song - Artist", "00:00 Song - Artist"). Focus on extracting the core "Song - Artist" pattern.
- **Fallback Strategy:** If the primary video is inaccessible (e.g., geo-blocked, age-restricted, transcript API blocked), use browser automation (`playwright_with_chunk`) as the reliable method to access the page and scrape the description.
- **Output Adherence:** Strictly follow the user-provided format template for the final output file.

## Common Triggers
Use this skill when the user request involves:
- "Find the YouTube video titled [X] and list the songs"
- "Extract the tracklist from this YouTube playlist/link"
- "Identify the names of each song from this YouTube music mix"
- Keywords: `YouTube playlist`, `song list`, `tracklist`, `music compilation`, `top hits`, `identify songs from lyrics`.

## Notes
- The primary data source is the video description, not audio analysis or transcripts.
- The skill assumes the tracklist is textually available in the description.
- Focus on accuracy of the extracted song and artist names over perfect timestamp alignment.
