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
echo "Downloading single plate dataset..."

huggingface-cli download altoslabs/scGeneScope measured/imaging/uint8/round_1/batch1/MIC1227_CP_Batch1_Plate1A_1/plate.zip --repo-type dataset --local-dir ./data/

# Now unzip all downloaded data
echo "All downloads complete. Now extracting files..."

unzip -o ./data/measured/imaging/uint8/round_1/batch1/MIC1227_CP_Batch1_Plate1A_1/plate.zip -d ./data/measured/imaging/uint8/round_1/batch1/MIC1227_CP_Batch1_Plate1A_1/
rm ./data/measured/imaging/uint8/round_1/batch1/MIC1227_CP_Batch1_Plate1A_1/plate.zip

# Last, download the manifest file
echo "Downloading manifest files..."

huggingface-cli download altoslabs/scGeneScope measured/imaging/round_1.h5 --repo-type dataset --local-dir ./data/
huggingface-cli download altoslabs/scGeneScope measured/imaging/round_2.h5 --repo-type dataset --local-dir ./data/

echo "Extraction complete. All data is now ready for use."