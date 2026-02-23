# Transit Line Query Reference

This document provides standardized search queries for fetching detailed transit line information, particularly station lists.

## General Pattern
For a given transit line and segment, construct queries to find:
1. The official line page (for comprehensive data).
2. The specific station list for a segment.

## Example Queries (Singapore MRT Context)

### For Line Overview & Station Lists
- `"[Line Name] Wikipedia"` (e.g., "Downtown Line Wikipedia")
- `"[Line Name] stations list"` (e.g., "East West Line stations list")
- `"[Line Name] route map"`

### For Segment Verification
- `"[Station A] to [Station B] [Line Name] stations"` (e.g., "Bugis to Tanah Merah East West Line stations")
- `"[Line Name] stations between [Station A] and [Station B]"`

### For Transfer Point Details
- `"[Station Name] MRT interchange"` (e.g., "Bugis MRT interchange lines")

## Preferred Sources (Priority Order)
1. **Wikipedia** - Usually contains complete station tables in chronological order.
2. **Official Transit Authority Websites** (e.g., Land Transport Authority (LTA) for Singapore) - For authoritative data.
3. **Reputable transit databases or forums** - As a fallback.

## Data Extraction Tips
- On Wikipedia, look for sections titled "Stations", "List of stations", or "Route". Station lists are often in tables or bulleted lists.
- Ensure the station order matches the direction of travel. Lines may have different station sequences for each direction.
- Note any branch lines (like the Changi Airport Branch) and treat them as separate segments.
