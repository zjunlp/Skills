# Conference Publication Data Sources

## Primary Sources (Official)
### CVPR (Computer Vision and Pattern Recognition)
- **Main site**: https://cvpr.thecvf.com/
- **Accepted papers**: https://cvpr.thecvf.com/Conferences/[YEAR]/AcceptedPapers
- **Open Access**: https://openaccess.thecvf.com/CVPR[YEAR]
- **Virtual site**: https://cvpr.thecvf.com/virtual/[YEAR]/papers.html

### NeurIPS (Neural Information Processing Systems)
- **Main site**: https://nips.cc/
- **Proceedings**: https://papers.nips.cc/
- **Current year**: https://papers.nips.cc/paper_files/paper/[YEAR]

### ICML (International Conference on Machine Learning)
- **Main site**: https://icml.cc/
- **Proceedings**: https://proceedings.mlr.press/
- **Current year**: https://proceedings.mlr.press/v[VOLUME]/

### ICLR (International Conference on Learning Representations)
- **Main site**: https://iclr.cc/
- **OpenReview**: https://openreview.net/group?id=ICLR.cc/[YEAR]/Conference

## Secondary/Aggregator Sources
### Paper Copilot
- **CVPR lists**: https://papercopilot.com/paper-list/cvpr-paper-list/cvpr-[YEAR]-paper-list/
- **Statistics**: https://papercopilot.com/statistics/cvpr-statistics/cvpr-[YEAR]-statistics

### Open Access Repositories
- **CVF Open Access**: https://openaccess.thecvf.com/
- **PMLR Proceedings**: https://proceedings.mlr.press/

### GitHub Collections
- **Top papers lists**: Often found on GitHub (e.g., SkalskiP/top-cvpr-2025-papers)
- **Community collections**: Search for "[CONFERENCE] [YEAR] papers GitHub"

## Search Strategies
### For finding paper lists:
- "[CONFERENCE] [YEAR] accepted papers"
- "[CONFERENCE] [YEAR] paper list"
- "[CONFERENCE] [YEAR] proceedings"

### For researcher information:
- "[Researcher Name] [University] homepage"
- "[Researcher Name] Google Scholar"
- "[University] [Department] faculty"

## Data Extraction Patterns
### From CVPR HTML:
- Paper titles: `<strong>` tags or `<a>` tags with links
- Authors: `<i>` tags within `indented` divs
- Affiliations: Usually in author lists, may need separate lookup
- Session info: "Poster Session X" or "Oral" tags

### From Paper Copilot:
- Metadata in JavaScript variables (look for `var meta = {...}`)
- Author statistics in structured format
- Affiliation data with counts

## Verification Sources
### Researcher Profiles:
- University faculty pages
- Google Scholar profiles
- Research lab websites
- LinkedIn (for current position)

### Institution Information:
- University department websites
- Campus location information
- Faculty directories
