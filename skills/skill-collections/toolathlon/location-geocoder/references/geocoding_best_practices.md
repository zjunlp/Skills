# Geocoding Best Practices

## Input Optimization

### 1. Location String Formatting
- **Always include country**: "Singapore Mobility Gallery, Singapore" not just "Singapore Mobility Gallery"
- **Use complete addresses** when possible: "1 Hampshire Rd, Block 1 Level 1, Singapore 219428"
- **Include landmarks** for ambiguous names: "Changi Airport MRT station" not just "Changi Airport"

### 2. Common Pitfalls to Avoid
- **Ambiguous names**: "Central Park" could be in New York, Sydney, or many other cities
- **Abbreviations**: Spell out street types (Road, Avenue, Boulevard)
- **Missing context**: "The Gallery" needs city/country context

## API Usage Guidelines

### 1. Rate Limiting Considerations
- Batch requests when processing multiple locations
- Implement exponential backoff for failed requests
- Cache frequently requested locations

### 2. Error Handling
- **No results**: Check spelling, add more context, try alternative phrasing
- **Multiple results**: Use the most specific match, ask user for clarification
- **Invalid requests**: Validate input format before API call

### 3. Response Processing
- Always check `formatted_address` matches intended location
- Verify coordinates are within expected geographic area
- Save `place_id` for future API calls to same location

## Accuracy Improvement Techniques

### 1. Pre-processing
