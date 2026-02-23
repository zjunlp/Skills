---
name: sports-schedule-analyzer
description: Analyzes sports league schedules to extract statistical insights about team schedules, including back-to-back games, home/away patterns, travel analysis, or schedule density. Processes schedule data from spreadsheets or structured sources, identifies consecutive game patterns, categorizes them by location configurations (HA, AH, HH, AA), and generates team-by-team breakdowns.
---
# Instructions

## 1. Identify and Load Schedule Data
- **Primary Source**: Look for a spreadsheet containing the schedule. The user may specify a title (e.g., "NHL 2425 Schedule") or you may need to list available spreadsheets to find it.
- **Confirm Structure**: The schedule should contain at minimum: **Date**, **Visitor Team**, and **Home Team** columns. Additional columns (time, scores, status) are acceptable but not required for core analysis.
- **Load Data**: Use the `google_sheet-get_sheet_data` tool to fetch the raw data. Be prepared for large datasets; the tool may return an "overlong" output saved to a file.

## 2. Parse and Prepare Data
- **Execute the Parsing Script**: Run the bundled `parse_schedule.py` script. It will:
    1. Load the overlong tool output from the workspace dump directory.
    2. Extract the `Date`, `Visitor`, and `Home` fields from each row (skipping the header).
    3. Build a chronological list of games for each team, tagging each game with its location (`H` for Home, `A` for Away).
    4. Sort each team's schedule by date.

## 3. Perform Back-to-Back Analysis
- **Execute the Analysis Script**: Run the bundled `analyze_back_to_back.py` script. It will:
    1. For each team, iterate through its sorted game list.
    2. Identify consecutive game dates (difference of 1 day).
    3. Categorize each back-to-back set into one of four configurations based on the location sequence:
        - **HA**: Home → Away
        - **AH**: Away → Home
        - **HH**: Home → Home
        - **AA**: Away → Away
    4. Tally the counts for each team and calculate the total.

## 4. Generate and Deliver Results
- **Create Output Spreadsheet**: Create a new Google Sheet with a descriptive title (e.g., "NHL-B2B-Analysis").
- **Rename Default Sheet**: Rename the default sheet to something meaningful like "B2B Analysis".
- **Populate Data**: Update the sheet with the results. The table must have the exact headers: `Team,HA,AH,HH,AA,Total`. Populate rows for all teams in alphabetical order.
- **Verify**: Optionally, fetch a small range of the new sheet to confirm successful creation and data accuracy.

## 5. Provide Summary Insights
After completing the analysis, offer a concise verbal summary to the user. Highlight:
- The team with the **most** and **fewest** total back-to-back sets.
- Notable patterns in configuration distribution (e.g., which configuration is most common league-wide).
- Any other observations from the data (e.g., teams with unusually high AA counts indicating long road trips).

## Key Triggers & Adaptations
- The skill is triggered by requests for: "back-to-back analysis", "schedule analysis", "home/away breakdown", "travel analysis", or analysis of "schedule patterns".
- While the example trajectory is for the NHL, the core logic applies to any league (NBA, MLB, NFL) with a similar schedule structure. Ensure the parsing script correctly identifies the relevant column names for the given dataset.
- If the user requests a different type of schedule analysis (e.g., longest homestand, travel miles), adapt the analysis script logic accordingly, using the parsed team schedule data as a foundation.
