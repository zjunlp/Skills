---
name: academic-researcher-matcher
description: When the user needs to find academic researchers or professors matching specific criteria (research field, institution location, publication metrics, conference activity). This skill analyzes user requirements, searches conference publication data (like CVPR, NeurIPS, ICML), identifies top authors by paper count, filters by research domain and institutional affiliation, and returns ranked matches. Triggers include 'find researchers', 'match professors', 'postdoctoral position', 'academic mentor', 'CVPR/NeurIPS/ICML papers', 'research field matching', 'university preference', 'publication count'.
---
# Skill: Academic Researcher Matcher

## Core Purpose
Help users identify top academic researchers (professors, postdoctoral supervisors) who match their specific career and research needs by analyzing recent high-impact conference publications.

## Primary Use Case
A doctoral student or early-career researcher seeking postdoctoral positions or academic mentors needs to find professors who:
1. Are active in their specific research field
2. Are affiliated with preferred institutions/locations
3. Have strong publication records at top conferences
4. Could be potential supervisors or collaborators

## Typical User Inputs
- Research field/area (e.g., "visual generative models", "diffusion models", "video generation")
- Location preferences (e.g., "Hong Kong universities", "US West Coast")
- Career stage (e.g., "postdoctoral position", "PhD supervisor")
- Conference focus (e.g., "CVPR 2025", "NeurIPS", "ICML")
- Output requirements (e.g., "top 3 matches", "save to file")

## Execution Workflow

### Phase 1: Understand User Requirements
1. **Read user's personal information** (if provided in workspace files like `personal_info.md`)
   - Extract research interests, career goals, location preferences
   - Note specific technical terms and keywords

2. **Clarify search criteria**
   - Conference/year (e.g., CVPR 2025)
   - Number of top researchers needed
   - Output format requirements
   - Any additional filters (institution type, professor rank, etc.)

### Phase 2: Gather Conference Publication Data
3. **Search for conference paper lists**
   - Use official conference websites (cvpr.thecvf.com, papers.nips.cc, etc.)
   - Alternative sources: papercopilot.com, Open Access repositories
   - Search queries: "[Conference] [Year] Accepted Papers"

4. **Extract structured data**
   - Paper titles, authors, affiliations
   - Session information, poster numbers
   - Author ordering and institutional affiliations

### Phase 3: Analyze and Filter Researchers
5. **Identify top authors by paper count**
   - Parse author lists from multiple papers
   - Count publications per researcher
   - Rank researchers by publication volume

6. **Filter by research domain**
   - Match paper titles/abstracts against user's research keywords
   - Focus on relevant subfields (e.g., diffusion models, video generation)
   - Exclude unrelated research areas

7. **Filter by institutional affiliation**
   - Check author affiliations against location preferences
   - Verify university/department information
   - Prioritize professors (Associate/Full Professor titles)

### Phase 4: Validate and Research Details
8. **Verify researcher profiles**
   - Search for professor homepages, Google Scholar profiles
   - Confirm research focus matches user interests
   - Check current position and availability

9. **Cross-reference with additional sources**
   - University department websites
   - Research lab pages
   - Recent publications beyond the target conference

### Phase 5: Present Results
10. **Compile final list**
    - Rank by match quality (publication count + research relevance + location match)
    - Include supporting evidence (paper counts, specific relevant papers)
    - Note any caveats or limitations

11. **Format output as requested**
    - Simple list format (e.g., one name per line)
    - More detailed profiles if requested
    - Save to specified file location

## Key Tools and Techniques
- **Web Search**: For finding conference paper lists and researcher profiles
- **HTML/Markdown Parsing**: To extract structured data from conference websites
- **Pattern Matching**: To identify research domain keywords in paper titles
- **Data Aggregation**: To count publications per researcher
- **Validation**: Cross-checking across multiple sources

## Common Challenges and Solutions
1. **Incomplete author lists**: Use multiple data sources, prioritize official conference pages
2. **Name disambiguation**: Check affiliations and research focus to distinguish researchers with common names
3. **Research domain matching**: Use specific technical terms from user's description
4. **Affiliation verification**: Check university websites for current faculty listings
5. **Data volume**: Start with sample data, then expand to full dataset as needed

## Output Examples
- Simple list: `Researcher Name\nResearcher Name\nResearcher Name`
- Detailed format: `Name (Institution) - X papers at [Conference] [Year] - Research: [Keywords]`

## Quality Checks
- Verify each researcher is actually a professor/supervisor (not student/postdoc)
- Confirm research alignment with user's stated interests
- Check that location preferences are met
- Ensure publication counts are accurate

## Notes for Implementation
- Be conservative in matching - better to have fewer high-quality matches than many poor ones
- When in doubt, prioritize researchers with clear professor titles and established labs
- Consider both publication quantity and research relevance
- Remember the user's ultimate goal: finding a suitable postdoctoral supervisor or academic mentor
