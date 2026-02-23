# Regex Pattern Reference for Privacy Desensitization

This document details the regular expression patterns used to identify sensitive information. Patterns must be applied in the specified order to avoid conflicts.

## Pattern Application Order

### 1. 1-800 Numbers (First to avoid partial matches)
- **Pattern**: `1-800-\d{3}-\d{4}`
- **Example**: `1-800-123-4567` → `/hidden/`
- **Pattern**: `1-800-[A-Z]+-?[A-Z]*`
- **Example**: `1-800-HOTEL-HELP` → `/hidden/`

### 2. Scientific Notation Credit Cards
- **Pattern**: `\d\.\d{2}E\+\d{2}`
- **Example**: `4.11E+15` → `/hidden/`
- **Note**: These appear in CSV exports and look similar to IP addresses.

### 3. Social Security Numbers (SSN)
- **Pattern**: `\b\d{3}-\d{2}-\d{4}\b`
- **Example**: `123-45-6789` → `/hidden/`
- **Note**: Word boundaries prevent matching partial strings.

### 4. Credit Card Numbers
- **16-digit with separators**: `\b\d{4}[-\s]\d{4}[-\s]\d{4}[-\s]\d{4}\b`
  - Example: `4111-1111-1111-1111` → `/hidden/`
- **Amex format (4-6-5)**: `\b\d{4}[-\s]\d{6}[-\s]\d{5}\b`
  - Example: `3782-822463-10005` → `/hidden/`
- **16 consecutive digits**: `\b\d{16}\b`
  - Example: `1234567890123456` → `/hidden/`

### 5. IP Addresses
- **Pattern**: `\b(?:\d{1,3}\.){3}\d{1,3}\b`
- **Example**: `192.168.1.100` → `/hidden/`
- **Note**: The pattern matches 0-999.0-999.0-999.0-999 but is sufficient for most cases.

### 6. Phone Numbers
- **Parentheses format**: `\(\d{3}\)\s*\d{3}[-.]?\d{4}`
  - Example: `(555) 123-4567` → `/hidden/`
- **Dash/dot format**: `\b\d{3}[-.]?\d{3}[-.]?\d{4}\b`
  - Example: `555-123-4567` or `555.123.4567` → `/hidden/`

### 7. Email Addresses
- **Pattern**: `[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}`
- **Example**: `user@example.com` → `/hidden/`

## Critical Implementation Notes

1. **Order Matters**: Applying patterns in the wrong order can cause:
   - Partial matches (e.g., `1-800-` being replaced, leaving `123-4567`)
   - False positives (e.g., dates being mistaken for IPs)

2. **Word Boundaries**: Use `\b` where appropriate to prevent matching substrings within larger numbers.

3. **Replacement String**: Always use exactly `/hidden/` as specified in requirements.

4. **Preservation**: The regex replacement should maintain surrounding text, punctuation, and formatting.

## Testing Patterns

Test the patterns against these edge cases:
- `Phone: (555) 123-4567 and 555-987-6543` → Should replace both
- `Date: 2024-12-01` → Should NOT be replaced (not SSN format)
- `Amount: $1,800.00` → Should NOT be replaced (not a phone number)
- `Network: 192.168.1.1 and 10.0.0.1` → Should replace both
- `Contact: support@company.com and admin@sub.domain.co.uk` → Should replace both
