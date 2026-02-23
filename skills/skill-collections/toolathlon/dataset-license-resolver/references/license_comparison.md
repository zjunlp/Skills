# License Comparison Guide

## Common License Types for AI/Data

### Permissive Licenses (Most to Least)
1. **MIT License** - Very permissive, allows almost any use with attribution
2. **Apache-2.0** - Similar to MIT but with patent protection
3. **BSD-3-Clause** - Requires attribution and disclaimer
4. **CC-BY-4.0** - Creative Commons Attribution, allows derivatives

### Restricted/Custom Licenses
- **DeepSeek License** - Use-based restrictions (see Attachment A for prohibited uses)
- **The Stack Terms of Use** - Requires attribution and compliance with original source licenses
- **Other (custom)** - Varies by project; requires reading specific terms

## Derivative Work Considerations

### Most Permissive for Derivatives:
1. MIT/Apache-2.0/BSD - Allow commercial use, modification, distribution
2. CC-BY - Allows derivatives with attribution
3. The Stack Terms - Allows derivatives but requires chain of attribution
4. DeepSeek License - Allows derivatives but with use restrictions

### Key Factors for Comparison:
1. **Attribution Requirements:** How must credit be given?
2. **Use Restrictions:** Are there prohibited applications?
3. **Commercial Use:** Is commercial use allowed?
4. **Redistribution:** Can the work be shared/modified?
5. **Patent Protection:** Does the license include patent grants?

## Investigation Checklist

### For GitHub Repositories:
1. Check for LICENSE, LICENSE.md, COPYING files
2. Examine README for license mentions
3. Check package configuration (setup.py, pyproject.toml)
4. Look for SPDX identifiers in source headers

### For Hugging Face Resources:
1. Check metadata tags (license:apache-2.0, license:other, etc.)
2. Read dataset/model card thoroughly
3. Look for "Terms of Use" or "License" sections
4. Check linked papers or documentation

### For Derived Works:
1. Trace provenance statements ("adopted from", "based on")
2. Identify original sources mentioned in documentation
3. Check for attribution requirements
4. Determine if the work is a direct copy, transformation, or synthesis
