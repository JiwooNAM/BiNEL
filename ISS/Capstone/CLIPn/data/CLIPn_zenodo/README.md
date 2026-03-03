# Datasets used in Scaling up functional prediction for uncharacterized small molecules through integration of diverse high-content screening resources

This dataset supports the development of CLIP<sup>n</sup>, a contrastive-learning framework designed to align heterogeneous high-content screening (HCS) profile datasets.

***GitHub link:*** https://github.com/AltschulerWu-Lab/CLIPn

## Directory Structure

### Data Files
- **HCS_datasets.pkl**: Contains 13 high-content screening (HCS) datasets from multiple studies across 20 years.
- **Hypoxia.pkl**: Contains 8 profile datasets using different assays and treated under diverse hypoxia durations.
- **Expression.pkl**: Contains 2 transcriptional profile datasets and 6 image profile datasets for multimodal analysis.

### Folders
#### raw_profiles
##### HCS13/
- Contains raw data from 13 high-content screening (HCS) datasets. Each dataset includes meta and feature files. 

##### L1000/
- **CDRP_feature_exp.csv**: Raw L1000 expression data from the CDRP dataset.
- **CDRP_meta_exp.csv**: Metadata associated with the CDRP expression data.
- **LINCS_feature_exp.csv**: Raw L1000 expression data from the LINCS dataset.
- **LINCS_meta_exp.csv**: Metadata associated with the LINCS expression data.

##### RxRx3/
- **RxRx3_feature_final.csv**: Profile data from the RxRx3 dataset.
- **RxRx3_meta_final.csv**: Metadata from the RxRx3 dataset.

##### Uncharacterized_compounds/
- **NCI_cpnData.csv**: Feature data for uncharacterized compounds from the NCI dataset.
- **NCI_cpnInfo.csv**: Information about uncharacterized compounds in the NCI dataset.
- **Prestwick_UTSW_cpnData.csv**: Feature data for uncharacterized compounds from the Prestwick UTSW dataset.
- **Prestwick_UTSW_cpnInfo.csv**: Information about uncharacterized compounds from the Prestwick UTSW dataset.


## Usage
```python
import pickle
with open('data.pkl', 'rb') as f:
    data = pickle.load(f)

X = data['X']
y = data['y']
```

## Data Reference
For raw datasets from 13 HCS database, data and analysis pipeline for dataset 1 was obtained from https://www.science.org/doi/suppl/10.1126/science.1100709/suppl_file/perlman.som.zip; for datasets 2-3, data were shared by authors; For datasets 4-5, analysis code was downloaded from https://static-content.springer.com/esm/art%3A10.1038%2Fnbt.3419/MediaObjects/41587_2016_BFnbt3419_MOESM21_ESM.zip and data were shared by authors; For datasets 6-7, processed dataset was downloaded from AWS following instructions from https://github.com/carpenter-singh-lab/2022_Haghighi_NatureMethods, and replicate_level_cp_normalized.csv.gz features were used. For project datasets 8-13, datasets and analysis results were downloaded from https://zenodo.org/records/7352487. For RxRx3, dataset was obtained from https://www.rxrx.ai/rxrx3. L1000 transcript datasets were downloaded using the same link as datasets 6-7 and the processed transcript data files (named “replicate_level_l1k.csv”) were used. 