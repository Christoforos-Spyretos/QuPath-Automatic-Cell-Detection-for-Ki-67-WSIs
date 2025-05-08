# Automatic Quantification of Ki-67 Labeling Index in Pediatric Brain Tumors Using QuPath

This repository includes an Apache Groovy script (Java-based syntax) for automated Ki-67 LI scoring, along with a Python script for post-processing to generate summary tables and graphical representations of the Ki-67 scores, and visualize density maps.

### Abstract
The quantification of the Ki-67 labeling index (LI) is critical for assessing tumor proliferation and prognosis in tumors, yet manual scoring remains a common practice. This study presents an automated workflow for Ki-67 scoring in whole slide images (WSIs) using an Apache Groovy code script for QuPath, complemented by a Python-based post-processing script, providing cell density maps and summary tables. The tissue and cell segmentation are performed using StarDist, a deep learning model, and adaptive thresholding to classify Ki-67 positive and negative nuclei. The pipeline was applied to a cohort of 632 pediatric brain tumor cases with 734 Ki-67-stained WSIs from the Children's Brain Tumor Network. Medulloblastoma showed the highest Ki-67 LI (median: 19.84, mean: 23.10 ± 16.15, maximum: 68.75), followed by atypical teratoid rhabdoid tumor (median: 19.36, mean: 20.48 ± 11.20). Moderate values were observed in brainstem glioma-diffuse intrinsic pontine glioma (median: 11.50), high-grade glioma (grades 3 & 4) (median: 9.50), and ependymoma (median: 5.88). Lower indices were found in meningioma (median: 1.84, mean: 3.37 ± 3.92), while the lowest were seen in low-grade glioma (grades 1 & 2) (median: 0.85), dysembryoplastic neuroepithelial tumor (median: 0.63), and ganglioglioma (median: 0.50). The results aligned with the consensus of the oncology, demonstrating a significant correlation in Ki-67 LI across most of the tumor families/types, with high malignancy tumors showing the highest proliferation indices and lower malignancy tumors exhibiting lower Ki-67 LI. The automated approach facilitates the assessment of large amounts of Ki-67 WSIs and offers adaptability for other immunohistochemical markers in research settings. 

## Table of Contents
- [Setup](#Setup)
- [Apache Groovy Script](#groovy)
- [Post Processing Python Script](#post-processing)
- [Reference](#reference)
- [License](#license)
---

## Setup

## Apache Groovy Script

## Post Processing Python Script

To run the post-processing Python script, open a terminal window, navigate to the directory containing all relevant files and folders, then execute the following command:
```bash
python summary_ratios.py --maps_dir /local/data1/chrsp39/QuPath_Portable/results --norm_maps_dir Normalized_Maps/ --result_dir Test_Results/ --data_dir /local/data1/chrsp39/QuPath_Portable/Project/data --csv_path /local/data1/chrsp39/QuPath_Portable/CBTN_KI67.csv --WSIs_path /local/data2/chrsp39/CBTN_v2/new_KI67/WSI
```

## Reference

## License
This work is licensed under [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International](https://creativecommons.org/licenses/by-nc-sa/4.0/)