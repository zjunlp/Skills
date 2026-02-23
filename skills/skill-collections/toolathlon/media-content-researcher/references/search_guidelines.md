# Media Search Guidelines

## Query Formulation
*   **For Movie Trailers:** `"[Exact Movie Title] official trailer YouTube"`
*   **For Music Videos:** `"[Artist Name] - [Song Title] official music video"`
*   **For TV Show Clips:** `"[Show Name] [Season] [Episode] clip official"`
*   **General Media:** `"[Content Name]" [platform, e.g., YouTube, Vimeo]`

## Source Verification Heuristics
1.  **Channel/Publisher Name:** Favor results from known, verified entities (e.g., `Marvel Entertainment`, `StarWars`, artist's VEVO channel).
2.  **Title Text:** Look for phrases like "Official Trailer", "Official Music Video", "Official Channel".
3.  **URL Patterns:** YouTube URLs from specific channels often follow a pattern (e.g., `youtube.com/user/Marvel`).
4.  **Snippet Text:** The description snippet in search results often confirms if the content is official.

## Result Parsing
*   Extract the **Title** and **Link** from the search tool's output.
*   The `snippet` field can provide useful context about the video's content and authenticity.
*   If the first result is not clearly official, examine subsequent results.
