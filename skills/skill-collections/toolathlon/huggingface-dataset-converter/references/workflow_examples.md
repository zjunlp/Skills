# Workflow Examples

## Example 1: DeepScaleR for Verl Framework

### User Request
"I need to download the most downloaded DeepScaleR dataset and convert it to verl format as verl_deepscaler.parquet"

### Agent Workflow
1. **Search**: Use `huggingface-dataset_search` with query="DeepScaleR", sort="downloads"
2. **Identify**: Select top result "agentica-org/DeepScaleR-Preview-Dataset"
3. **Read Format**: Load `/workspace/dumps/workspace/format.json`
4. **Convert**: Run conversion script with:
   