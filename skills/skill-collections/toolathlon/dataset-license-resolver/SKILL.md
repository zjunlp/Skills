---
name: dataset-license-resolver
description: When the user needs to determine and document licenses for datasets or AI models, particularly when dealing with derivative works or synthesized content. This skill handles 1) Identifying direct data sources and models used in dataset creation, 2) Researching original licenses from upstream repositories (GitHub, Hugging Face), 3) Comparing license permissiveness for derivative/secondary use, 4) Applying the most appropriate license based on user requirements, 5) Updating documentation and repository metadata with license information. Triggers include license inquiries, dataset attribution requests, compliance documentation needs, repository metadata updates, and derivative work licensing questions.
---
# Instructions

## 1. Understand the Request
- Parse the user's request to identify:
  - The specific datasets/models in question
  - The user's license selection criteria (e.g., "most permissive for derivative use")
  - Required output format (e.g., issue reply, README updates)
  - Any authentication tokens needed (e.g., `.hf_token`)

## 2. Investigate Data Sources
**For each dataset/model mentioned:**
- **GitHub Repositories:**
  - Use `github-get_file_contents` to examine repository structure and README
  - Look for license files (LICENSE, LICENSE.md, etc.) and documentation
- **Hugging Face Resources:**
  - Use `huggingface-hub_repo_details` to get metadata including license tags
  - Use `fetch-fetch_markdown` to retrieve full README/content
  - Search for related datasets/models using `huggingface-dataset_search` or `huggingface-model_search`
- **Trace Provenance:**
  - Follow references to original sources (e.g., "adopted from HuggingFaceTB team")
  - Identify base models used for synthesis (e.g., "uses DeepSeek-V2.5")

## 3. Determine License Hierarchy
**When multiple sources exist:**
1. **Identify direct sources:** For raw/derived datasets, find the original data source
2. **Identify synthesis models:** For processed/transformed datasets, find the models used for generation
3. **Compare licenses:** Evaluate which is more permissive for derivative/secondary use based on:
   - Attribution requirements
   - Use-based restrictions
   - Commercial use allowances
   - Redistribution terms

## 4. Apply License Selection Logic
**Follow user's priority rules:**
- If user specifies "most permissive for derivative/secondary use," compare licenses and select the one with fewer restrictions
- If user specifies "directly reuse license from source," use the identified source's license
- Document the reasoning for license selection

## 5. Execute Required Actions
**Based on user requirements:**
- **Reply to Issues:**
  - Format response exactly as specified by user
  - Include all required placeholders filled with determined licenses
- **Update Documentation:**
  - Retrieve current README/content using appropriate tools
  - Append license section at the end as specified
  - Maintain existing formatting and content
- **Authentication:**
  - Use provided tokens (e.g., read `.hf_token` file)
  - Pass tokens to API calls as needed

## 6. Complete and Close
- Verify all actions completed successfully
- Provide confirmation of updates made
- Close issues if requested
- Claim task completion

## Key Considerations
- **License Types:** Be familiar with common licenses (MIT, Apache-2.0, CC-BY, etc.) and custom licenses (DeepSeek License, The Stack Terms of Use)
- **Provenance Chains:** Some datasets may have multiple layers of derivation; trace back to the original source
- **Format Compliance:** Strictly adhere to user-specified output formats; do not modify, add, or remove content outside placeholders
- **Error Handling:** If license information cannot be determined, document the investigation and ask for clarification
