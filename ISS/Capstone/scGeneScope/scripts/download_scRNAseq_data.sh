#! /bin/bash
# This script downloads and unzips imaging data from the Hugging Face Hub
# It retrieves multiple compressed plate datasets from the altoslabs/scGeneScope repository
# and extracts them while maintaining the original directory structure.
# The script requires a Hugging Face token (HF_TOKEN) to be set as an environment variable.

# Check if HF_TOKEN is defined
if [ -z "$HF_TOKEN" ]; then
    echo "Error: HF_TOKEN environment variable is not set."
    echo "Please set your Hugging Face token by running:"
    echo "export HF_TOKEN=your_huggingface_token"
    exit 1
fi

echo "Hugging Face token found. Proceeding with downloads..."

# First, download all data from Hugging Face Hub
echo "Downloading all datasets..."

huggingface-cli download altoslabs/scGeneScope measured/rnaseq/round_1.h5ad --repo-type dataset --local-dir ./data/
huggingface-cli download altoslabs/scGeneScope measured/rnaseq/round_2.h5ad --repo-type dataset --local-dir ./data/