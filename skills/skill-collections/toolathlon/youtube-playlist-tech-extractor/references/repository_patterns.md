# Common GitHub Repository URL Patterns
When searching for a technology's GitHub repository, try these patterns in order.

## Pattern 1: Organization/Project
The most common pattern. The organization is often the company or main research group.
*   `https://github.com/anthropics/claude-code`
*   `https://github.com/Dao-AILab/flash-attention`
*   `https://github.com/QwenLM/Qwen`
*   `https://github.com/mistralai/mistral-inference`
*   `https://github.com/google-gemini/gemini-cli`

## Pattern 2: ProjectName/ProjectName
Some projects have an organization named after the project itself.
*   `https://github.com/OpenHands/OpenHands`

## Pattern 3: Features Page (Proprietary)
For commercial products, the main page may be under `github.com/features`.
*   `https://github.com/features/copilot`

## Search Strategy
1.  Extract the clean project name from the video title (e.g., "Claude Code" -> `claude-code`).
2.  Identify the likely organization (e.g., "Claude Code" is by Anthropic -> `anthropics`).
3.  Construct URL: `https://github.com/<organization>/<project-name>`.
4.  Navigate. If 404, the project may be closed source, or the organization/project name may be different. Use your judgment or note it as proprietary.
