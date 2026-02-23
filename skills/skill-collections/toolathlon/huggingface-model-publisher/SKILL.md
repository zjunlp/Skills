---
name: huggingface-model-publisher
description: When the user wants to publish a trained model to Hugging Face Hub with proper documentation and metadata. This skill handles authentication with Hugging Face using API tokens, creates or updates model repositories, uploads model files (config, weights, tokenizers), generates comprehensive README.md files with benchmark results, and ensures all necessary files are properly structured for the Hugging Face ecosystem. It's triggered when users mention 'push to Hugging Face', 'upload model to HF Hub', or need to share models with proper documentation.
---
# Instructions

## Overview
This skill publishes a trained model checkpoint to Hugging Face Hub. It identifies the best model based on evaluation metrics, prepares the model folder with necessary documentation, and uploads everything to a specified repository.

## Prerequisites
1. **Hugging Face Token**: A valid Hugging Face API token must be available, typically in a file like `hf_token.txt` in the workspace.
2. **Model Checkpoints**: Trained model checkpoints organized in directories (e.g., `checkpoints/step_100`, `checkpoints/step_200`).
3. **Evaluation System**: Benchmark evaluation scripts that can calculate scores for different model checkpoints.

## Step-by-Step Process

### 1. Scan Workspace and Identify Best Model
- List all available model checkpoints in the workspace.
- For each checkpoint, run evaluation across all benchmark categories.
- Calculate an overall `eval_accuracy` score (typically the average of all benchmark scores).
- Select the checkpoint with the highest `eval_accuracy`.

**Key Decision Points:**
- The evaluation logic may be implemented in compiled Cython modules. If they cannot be imported directly, you may need to reimplement the scoring formulas based on available source code or documentation.
- Benchmark categories typically include: math_reasoning, logical_reasoning, common_sense, reading_comprehension, question_answering, text_classification, sentiment_analysis, code_generation, creative_writing, dialogue_generation, summarization, translation, knowledge_retrieval, instruction_following, safety_evaluation.

### 2. Prepare Model Folder
- Copy the selected checkpoint folder to a temporary location if needed.
- Ensure the folder contains at minimum:
  - `config.json`: Model configuration
  - `pytorch_model.bin`: Model weights
  - Any tokenizer files if applicable
- Add a comprehensive `README.md` file with:
  - Model metadata (license, library_name)
  - Introduction and model description
  - **Complete benchmark results table** with scores for all evaluated categories (keep three decimal places)
  - Usage instructions, system prompts, temperature recommendations
  - License information
  - Contact details
- Include any asset files (e.g., figures referenced in the README).

### 3. Authenticate with Hugging Face Hub
- Read the Hugging Face API token from the workspace.
- Use `huggingface_hub.login()` or equivalent to authenticate.
- Verify authentication by checking `api.whoami()`.

### 4. Create/Update Repository
- Determine the target repository name (usually specified by user or derived from model name).
- Create the repository using `create_repo()` with `exist_ok=True` to handle existing repositories.
- The repository should be of type `"model"`.

### 5. Upload Model Files
- Upload the entire prepared model folder using `api.upload_folder()`.
- Ensure all files are uploaded: README.md, config.json, pytorch_model.bin, tokenizer files, and assets.
- Verify the upload by listing repository files.

### 6. Final Verification
- Download the README.md from the repository to confirm benchmark scores were correctly uploaded.
- Check that all necessary files are present in the repository.
- Provide the user with the model URL: `https://huggingface.co/<username>/<repo_name>`

## Error Handling
- **Missing Token**: If `hf_token.txt` is not found, ask the user to provide it or check the workspace.
- **Invalid Checkpoints**: If no valid checkpoints are found, inform the user and suggest checking the workspace structure.
- **Upload Failures**: If upload fails, check network connectivity, token permissions, and repository visibility settings.
- **Benchmark Evaluation Failures**: If evaluation scripts fail, consider implementing fallback scoring based on available formulas.

## Notes
- The skill should preserve all original README.md content except for updating the benchmark scores table.
- Always keep benchmark scores to three decimal places for consistency.
- The model should be properly tagged with appropriate metadata (license, library_name, etc.) in the README frontmatter.
