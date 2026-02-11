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

huggingface-cli download altoslabs/scGeneScope measured/imaging/uint8/round_1/batch1/MIC1227_CP_Batch1_Plate1A_1/plate.zip --repo-type dataset --local-dir ./data
huggingface-cli download altoslabs/scGeneScope measured/imaging/uint8/round_1/batch1/MIC1227_CP_Batch1_Plate1A_2/plate.zip --repo-type dataset --local-dir ./data
huggingface-cli download altoslabs/scGeneScope measured/imaging/uint8/round_1/batch1/MIC1227_CP_Batch1_Plate1A_3/plate.zip --repo-type dataset --local-dir ./data
huggingface-cli download altoslabs/scGeneScope measured/imaging/uint8/round_1/batch1/MIC1227_CP_Batch1_Plate1B_1/plate.zip --repo-type dataset --local-dir ./data
huggingface-cli download altoslabs/scGeneScope measured/imaging/uint8/round_1/batch2/MIC1227_CP_Batch2_Plate2A_1/plate.zip --repo-type dataset --local-dir ./data
huggingface-cli download altoslabs/scGeneScope measured/imaging/uint8/round_1/batch2/MIC1227_CP_Batch2_Plate2B_1/plate.zip --repo-type dataset --local-dir ./data
huggingface-cli download altoslabs/scGeneScope measured/imaging/uint8/round_1/batch3/MIC1227_CP_Batch3_Plate3A_1/plate.zip --repo-type dataset --local-dir ./data
huggingface-cli download altoslabs/scGeneScope measured/imaging/uint8/round_1/batch3/MIC1227_CP_Batch3_Plate3B_1/plate.zip --repo-type dataset --local-dir ./data
huggingface-cli download altoslabs/scGeneScope measured/imaging/uint8/round_1/batch3/MIC1227_CP_Batch3_Plate3B_2/plate.zip --repo-type dataset --local-dir ./data
huggingface-cli download altoslabs/scGeneScope measured/imaging/uint8/round_1/batch4/MIC1227_CP_Batch4_4B_1/plate.zip --repo-type dataset --local-dir ./data
huggingface-cli download altoslabs/scGeneScope measured/imaging/uint8/round_1/batch4/MIC1227_CP_Batch4_Plate4A_1/plate.zip --repo-type dataset --local-dir ./data
huggingface-cli download altoslabs/scGeneScope measured/imaging/uint8/round_1/batch4/MIC1227_CP_Batch4_Plate4B_1/plate.zip --repo-type dataset --local-dir ./data
huggingface-cli download altoslabs/scGeneScope measured/imaging/uint8/round_1/batch5/MIC1227_CP_Batch5_5A_1/plate.zip --repo-type dataset --local-dir ./data
huggingface-cli download altoslabs/scGeneScope measured/imaging/uint8/round_1/batch5/MIC1227_CP_Batch5_5B_1/plate.zip --repo-type dataset --local-dir ./data
huggingface-cli download altoslabs/scGeneScope measured/imaging/uint8/round_1/batch6/MIC1227_CP_Batch6_6A_1/plate.zip --repo-type dataset --local-dir ./data
huggingface-cli download altoslabs/scGeneScope measured/imaging/uint8/round_1/batch6/MIC1227_CP_Batch6_6B_1/plate.zip --repo-type dataset --local-dir ./data
huggingface-cli download altoslabs/scGeneScope measured/imaging/uint8/round_2/batch1/240110_PA_Plate5_CP/plate.zip --repo-type dataset --local-dir ./data
huggingface-cli download altoslabs/scGeneScope measured/imaging/uint8/round_2/batch1/240110_PA_Plate6_CP/plate.zip --repo-type dataset --local-dir ./data
huggingface-cli download altoslabs/scGeneScope measured/imaging/uint8/round_2/batch2/20240117_PA_batch2_Plate11cp/plate.zip --repo-type dataset --local-dir ./data
huggingface-cli download altoslabs/scGeneScope measured/imaging/uint8/round_2/batch2/20240117_PA_batch2_Plate11cp_2/plate.zip --repo-type dataset --local-dir ./data
huggingface-cli download altoslabs/scGeneScope measured/imaging/uint8/round_2/batch2/20240117_PA_batch2_Plate12cp/plate.zip --repo-type dataset --local-dir ./data
huggingface-cli download altoslabs/scGeneScope measured/imaging/uint8/round_2/batch2/20240227_PA_batch2_Plate11cp_B03/plate.zip --repo-type dataset --local-dir ./data
huggingface-cli download altoslabs/scGeneScope measured/imaging/uint8/round_2/batch3/20240125_PA_batch3_Plate17cp/plate.zip --repo-type dataset --local-dir ./data
huggingface-cli download altoslabs/scGeneScope measured/imaging/uint8/round_2/batch3/20240125_PA_batch3_Plate18cp/plate.zip --repo-type dataset --local-dir ./data
huggingface-cli download altoslabs/scGeneScope measured/imaging/uint8/round_2/batch4/20240306_PA_batch4_Plate23cp/plate.zip --repo-type dataset --local-dir ./data
huggingface-cli download altoslabs/scGeneScope measured/imaging/uint8/round_2/batch4/20240306_PA_batch4_Plate24cp/plate.zip --repo-type dataset --local-dir ./data

# Now unzip all downloaded data
echo "All downloads complete. Now extracting files..."

unzip -o ./data/measured/imaging/uint8/round_1/batch1/MIC1227_CP_Batch1_Plate1A_1/plate.zip -d ./data/measured/imaging/uint8/round_1/batch1/MIC1227_CP_Batch1_Plate1A_1/
unzip -o ./data/measured/imaging/uint8/round_1/batch1/MIC1227_CP_Batch1_Plate1A_2/plate.zip -d ./data/measured/imaging/uint8/round_1/batch1/MIC1227_CP_Batch1_Plate1A_2/
unzip -o ./data/measured/imaging/uint8/round_1/batch1/MIC1227_CP_Batch1_Plate1A_3/plate.zip -d ./data/measured/imaging/uint8/round_1/batch1/MIC1227_CP_Batch1_Plate1A_3/
unzip -o ./data/measured/imaging/uint8/round_1/batch1/MIC1227_CP_Batch1_Plate1B_1/plate.zip -d ./data/measured/imaging/uint8/round_1/batch1/MIC1227_CP_Batch1_Plate1B_1/
unzip -o ./data/measured/imaging/uint8/round_1/batch2/MIC1227_CP_Batch2_Plate2A_1/plate.zip -d ./data/measured/imaging/uint8/round_1/batch2/MIC1227_CP_Batch2_Plate2A_1/
unzip -o ./data/measured/imaging/uint8/round_1/batch2/MIC1227_CP_Batch2_Plate2B_1/plate.zip -d ./data/measured/imaging/uint8/round_1/batch2/MIC1227_CP_Batch2_Plate2B_1/
unzip -o ./data/measured/imaging/uint8/round_1/batch3/MIC1227_CP_Batch3_Plate3A_1/plate.zip -d ./data/measured/imaging/uint8/round_1/batch3/MIC1227_CP_Batch3_Plate3A_1/
unzip -o ./data/measured/imaging/uint8/round_1/batch3/MIC1227_CP_Batch3_Plate3B_1/plate.zip -d ./data/measured/imaging/uint8/round_1/batch3/MIC1227_CP_Batch3_Plate3B_1/
unzip -o ./data/measured/imaging/uint8/round_1/batch3/MIC1227_CP_Batch3_Plate3B_2/plate.zip -d ./data/measured/imaging/uint8/round_1/batch3/MIC1227_CP_Batch3_Plate3B_2/
unzip -o ./data/measured/imaging/uint8/round_1/batch4/MIC1227_CP_Batch4_4B_1/plate.zip -d ./data/measured/imaging/uint8/round_1/batch4/MIC1227_CP_Batch4_4B_1/
unzip -o ./data/measured/imaging/uint8/round_1/batch4/MIC1227_CP_Batch4_Plate4A_1/plate.zip -d ./data/measured/imaging/uint8/round_1/batch4/MIC1227_CP_Batch4_Plate4A_1/
unzip -o ./data/measured/imaging/uint8/round_1/batch4/MIC1227_CP_Batch4_Plate4B_1/plate.zip -d ./data/measured/imaging/uint8/round_1/batch4/MIC1227_CP_Batch4_Plate4B_1/
unzip -o ./data/measured/imaging/uint8/round_1/batch5/MIC1227_CP_Batch5_5A_1/plate.zip -d ./data/measured/imaging/uint8/round_1/batch5/MIC1227_CP_Batch5_5A_1/
unzip -o ./data/measured/imaging/uint8/round_1/batch5/MIC1227_CP_Batch5_5B_1/plate.zip -d ./data/measured/imaging/uint8/round_1/batch5/MIC1227_CP_Batch5_5B_1/
unzip -o ./data/measured/imaging/uint8/round_1/batch6/MIC1227_CP_Batch6_6A_1/plate.zip -d ./data/measured/imaging/uint8/round_1/batch6/MIC1227_CP_Batch6_6A_1/
unzip -o ./data/measured/imaging/uint8/round_1/batch6/MIC1227_CP_Batch6_6B_1/plate.zip -d ./data/measured/imaging/uint8/round_1/batch6/MIC1227_CP_Batch6_6B_1/
unzip -o ./data/measured/imaging/uint8/round_2/batch1/240110_PA_Plate5_CP/plate.zip -d ./data/measured/imaging/uint8/round_2/batch1/240110_PA_Plate5_CP/
unzip -o ./data/measured/imaging/uint8/round_2/batch1/240110_PA_Plate6_CP/plate.zip -d ./data/measured/imaging/uint8/round_2/batch1/240110_PA_Plate6_CP/
unzip -o ./data/measured/imaging/uint8/round_2/batch2/20240117_PA_batch2_Plate11cp/plate.zip -d ./data/measured/imaging/uint8/round_2/batch2/20240117_PA_batch2_Plate11cp/
unzip -o ./data/measured/imaging/uint8/round_2/batch2/20240117_PA_batch2_Plate11cp_2/plate.zip -d ./data/measured/imaging/uint8/round_2/batch2/20240117_PA_batch2_Plate11cp_2/
unzip -o ./data/measured/imaging/uint8/round_2/batch2/20240117_PA_batch2_Plate12cp/plate.zip -d ./data/measured/imaging/uint8/round_2/batch2/20240117_PA_batch2_Plate12cp/
unzip -o ./data/measured/imaging/uint8/round_2/batch2/20240227_PA_batch2_Plate11cp_B03/plate.zip -d ./data/measured/imaging/uint8/round_2/batch2/20240227_PA_batch2_Plate11cp_B03/
unzip -o ./data/measured/imaging/uint8/round_2/batch3/20240125_PA_batch3_Plate17cp/plate.zip -d ./data/measured/imaging/uint8/round_2/batch3/20240125_PA_batch3_Plate17cp/
unzip -o ./data/measured/imaging/uint8/round_2/batch3/20240125_PA_batch3_Plate18cp/plate.zip -d ./data/measured/imaging/uint8/round_2/batch3/20240125_PA_batch3_Plate18cp/
unzip -o ./data/measured/imaging/uint8/round_2/batch4/20240306_PA_batch4_Plate23cp/plate.zip -d ./data/measured/imaging/uint8/round_2/batch4/20240306_PA_batch4_Plate23cp/
unzip -o ./data/measured/imaging/uint8/round_2/batch4/20240306_PA_batch4_Plate24cp/plate.zip -d ./data/measured/imaging/uint8/round_2/batch4/20240306_PA_batch4_Plate24cp/

# remove the zip files
rm ./data/measured/imaging/uint8/round_1/batch1/MIC1227_CP_Batch1_Plate1A_1/plate.zip
rm ./data/measured/imaging/uint8/round_1/batch1/MIC1227_CP_Batch1_Plate1A_2/plate.zip
rm ./data/measured/imaging/uint8/round_1/batch1/MIC1227_CP_Batch1_Plate1A_3/plate.zip
rm ./data/measured/imaging/uint8/round_1/batch1/MIC1227_CP_Batch1_Plate1B_1/plate.zip
rm ./data/measured/imaging/uint8/round_1/batch2/MIC1227_CP_Batch2_Plate2A_1/plate.zip
rm ./data/measured/imaging/uint8/round_1/batch2/MIC1227_CP_Batch2_Plate2B_1/plate.zip
rm ./data/measured/imaging/uint8/round_1/batch3/MIC1227_CP_Batch3_Plate3A_1/plate.zip
rm ./data/measured/imaging/uint8/round_1/batch3/MIC1227_CP_Batch3_Plate3B_1/plate.zip
rm ./data/measured/imaging/uint8/round_1/batch3/MIC1227_CP_Batch3_Plate3B_2/plate.zip
rm ./data/measured/imaging/uint8/round_1/batch4/MIC1227_CP_Batch4_4B_1/plate.zip
rm ./data/measured/imaging/uint8/round_1/batch4/MIC1227_CP_Batch4_Plate4A_1/plate.zip
rm ./data/measured/imaging/uint8/round_1/batch4/MIC1227_CP_Batch4_Plate4B_1/plate.zip
rm ./data/measured/imaging/uint8/round_1/batch5/MIC1227_CP_Batch5_5A_1/plate.zip
rm ./data/measured/imaging/uint8/round_1/batch5/MIC1227_CP_Batch5_5B_1/plate.zip
rm ./data/measured/imaging/uint8/round_1/batch6/MIC1227_CP_Batch6_6A_1/plate.zip
rm ./data/measured/imaging/uint8/round_1/batch6/MIC1227_CP_Batch6_6B_1/plate.zip
rm ./data/measured/imaging/uint8/round_2/batch1/240110_PA_Plate5_CP/plate.zip
rm ./data/measured/imaging/uint8/round_2/batch1/240110_PA_Plate6_CP/plate.zip
rm ./data/measured/imaging/uint8/round_2/batch2/20240117_PA_batch2_Plate11cp/plate.zip
rm ./data/measured/imaging/uint8/round_2/batch2/20240117_PA_batch2_Plate11cp_2/plate.zip
rm ./data/measured/imaging/uint8/round_2/batch2/20240117_PA_batch2_Plate12cp/plate.zip
rm ./data/measured/imaging/uint8/round_2/batch2/20240227_PA_batch2_Plate11cp_B03/plate.zip
rm ./data/measured/imaging/uint8/round_2/batch3/20240125_PA_batch3_Plate17cp/plate.zip
rm ./data/measured/imaging/uint8/round_2/batch3/20240125_PA_batch3_Plate18cp/plate.zip
rm ./data/measured/imaging/uint8/round_2/batch4/20240306_PA_batch4_Plate23cp/plate.zip
rm ./data/measured/imaging/uint8/round_2/batch4/20240306_PA_batch4_Plate24cp/plate.zip

# Last, download the manifest file
echo "Downloading manifest files..."

huggingface-cli download altoslabs/scGeneScope measured/imaging/round_1.h5 --repo-type dataset --local-dir ./data/
huggingface-cli download altoslabs/scGeneScope measured/imaging/round_2.h5 --repo-type dataset --local-dir ./data/

echo "Extraction complete. All data is now ready for use."