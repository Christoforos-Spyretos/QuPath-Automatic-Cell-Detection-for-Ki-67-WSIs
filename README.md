# Automatic Quantification of Ki-67 Labeling Index in Pediatric Brain Tumors Using QuPath

This repository includes an Apache Groovy script (Java-based syntax) for automated Ki-67 LI scoring, along with a Python script for post-processing to generate summary tables and graphical representations of the Ki-67 scores, and visualize density maps.

[Preprint](https://doi.org/10.1101/2025.05.09.25327292) | [Cite](#reference)

## Abstract   
The quantification of the Ki-67 labeling index (LI) is critical for assessing tumor proliferation and prognosis in tumors, yet manual scoring remains a common practice. This study presents an automated workflow for Ki-67 scoring in whole slide images (WSIs) using an Apache Groovy code script for QuPath, complemented by a Python-based post-processing script, providing cell density maps and summary tables. The tissue and cell segmentation are performed using StarDist, a deep learning model, and adaptive thresholding to classify Ki-67 positive and negative nuclei. The pipeline was applied to a cohort of 632 pediatric brain tumor cases with 734 Ki-67-stained WSIs from the Children's Brain Tumor Network. Medulloblastoma showed the highest Ki-67 LI (median: 19.84, mean: 23.10 ± 16.15, maximum: 68.75), followed by atypical teratoid rhabdoid tumor (median: 19.36, mean: 20.48 ± 11.20). Moderate values were observed in brainstem glioma-diffuse intrinsic pontine glioma (median: 11.50), high-grade glioma (grades 3 & 4) (median: 9.50), and ependymoma (median: 5.88). Lower indices were found in meningioma (median: 1.84, mean: 3.37 ± 3.92), while the lowest were seen in low-grade glioma (grades 1 & 2) (median: 0.85), dysembryoplastic neuroepithelial tumor (median: 0.63), and ganglioglioma (median: 0.50). The results aligned with the consensus of the oncology, demonstrating a significant correlation in Ki-67 LI across most of the tumor families/types, with high malignancy tumors showing the highest proliferation indices and lower malignancy tumors exhibiting lower Ki-67 LI. The automated approach facilitates the assessment of large amounts of Ki-67 WSIs and offers adaptability for other immunohistochemical markers in research settings. 

## Table of Contents
- [Setup](#Setup)
- [Apache Groovy Script](#groovy)
- [Post Processing Python Script](#post-processing)
- [Reference](#reference)
- [License](#license)
---

## Setup

1. **Install QuPath**  
   Download and install [QuPath](https://qupath.github.io), an open-source software platform for digital pathology image analysis, from its official website.

2. **Clone the Repository**  
   Clone this repository by following the official GitHub instructions on [how to clone a repository](https://docs.github.com/en/repositories/creating-and-managing-repositories/cloning-a-repository).

3. **Create a Project Folder**  
   Inside the cloned repository, create a new folder to serve as your QuPath project workspace. For example, it can be named *Ki-67 Project*.

4. **Prepare Metadata**  
   Create a `.csv` file with the following headers:  
   - `case_id`: unique id for each patient or case.  
   - `slide_id`: id for the whole slide image.  
   - `label`: label used for grouping or classification (e.g., tumor type).  

   Below is an example of how the `.csv` file should look:  
   | case_id   | slide_id   | label   |
   |-----------|------------|---------|
   | case_001  | slide_001  | label_1 |
   | case_002  | slide_002  | label_2 |
   | case_003  | slide_003  | label_1 |

   If either `case_id` or `slide_id` is unavailable, the same value can be used for both fields.

5. **Install Miniconda**    
   Download  and install [Miniconda](https://docs.conda.io/projects/conda/en/latest/index.html) from the official website, it is needed to be able to run the python post-processing script.

6. **Create a Conda Environment**  
   Open a terminal window and create a conda environment following the [instructions](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) from the official website. For example run in the terminal `conda create -n qupath python=3.12`.

7. **Activate the Environment**  
   Activate the conda environment executing the command `conda activate qupath`.

8. **Install Required Dependencies**
   In the terminal, navigate to the cloned repository directory, then execute the command `pip install -r requirements.txt` to download the required libraries to be able to run the python post-processing script.

## Apache Groovy Script

To run the Apache Groovy script for automated Ki-67 labeling index (LI) scoring, follow the steps below:

1. **Open QuPath**  
   Launch the QuPath application on your computer.

2. **Create a New Project**  
   In QuPath, navigate to:  
   `File` -> `Project...` -> `Create project`  
   A window will appear prompting to select a directory. Choose the folder that will serve as the QuPath project workspace (e.g., Ki-67 Project).

3. **Add Images to the Project**  
   In QuPath, navigate to:  
   `File` -> `Project...` -> `Add images`  
   A file browser window will open, either drag and drop the WSIs into this window or navigate through the filesystem to import them. Once imported, the images will appear in the project panel on the left-hand side of the QuPath interface.

4. **Import the StarDist Extension**  
   Navigate to the directory where the repository is cloned, then to QP Extensions -> extensions. Drag and drop the *qupath-extension-stardist-0.5.0.jar* file into the QuPath window. It will be asked to set a folder to save the program extensions, which can be the same folder where the file was dragged from.

5. **Run the Project Script**  
   To execute the Groovy script, navigate in the QuPath window to:  
   `Automate` -> `Shared scripts` -> `Pos_cellD_QuPath_project`  
   This will open the Script Editor window.
   - To process a single image, click `Run`, and relevant output will be shown in the Script Editor. Note that the results must be saved manually before closing QuPath.
   - To process multiple images, click the three dots next to Run, choose Run for project, select the images to analyze, and click OK. A progress window will appear, and relevant output will be shown in the Script Editor.

6. **Review Results**  
   After processing, tissue segmentation, cell segmentation, and classification can be reviewed directly in QuPath interface. Additionally, a data folder will be created in the project's directory (e.g., Ki-67 Project), in which the output files generated by the Pos_cellD_QuPath_project script are stored. A results folder will also be generated, containing:
   - an Area_Det.txt file, which includes the annotation area (in mm²) and total cell count per WSI.
   - a Raw Density Maps folder, which stores the raw cell density map images created during analysis.

## Post Processing Python Script
To run the post-processing Python script follow the steps below:

1. Open a terminal window.

2. Activate the conda environment executing the command `conda activate qupath`.

3. Navigate to the directory, in which the repository is cloned.

4. Then execute the following command:
    ```bash
    python summary_ratios.py --maps_dir MAPS_DIRECTORY --data_dir PROJECT_DATA_DIRECTORY --csv_path PATH_to_CSV --WSIs_dir WSI_DIRECTORY --norm_maps_dir NORM_MAPS_DIRECTORY --result_dir RESULT_DIRECTORY
    ```
    Arguments:
    - `--maps_dir`: Directory of the cell density maps generated by QuPath.
    - `--data_dir`: Directory of the project data.
    - `--csv_path`: Path to the CSV file containing WSI metadata.
    - `--WSIs_dir`: Directory of the folder where the WSIs exist.
    - `--norm_maps_dir`: Directory where normalized density maps will be saved.
    - `--result_dir`: Directory where the summary tables and graphs will be saved.
   
   Once executed, the script will display progress in the terminal. It will generate:
   - a folder with the normalised positive and negative cell density, and Ki-67 LI maps.
   - a folder with summary graphs and tables will be generated.

## Reference

```
@article {Spyretos2025.05.09.25327292,
	author = {Spyretos, Christoforos and Pardo Ladino, Juan Manuel and Blomstrand, Hakon and Nyman, Per and Snodahl, Oscar and Shamikh, Alia and Elander, Nils O. and Haj-Hosseini, Neda},
	title = {Automatic Quantification of Ki-67 Labeling Index in Pediatric Brain Tumors Using QuPath},
	elocation-id = {2025.05.09.25327292},
	year = {2025},
	doi = {10.1101/2025.05.09.25327292},
	publisher = {Cold Spring Harbor Laboratory Press},
	URL = {https://www.medrxiv.org/content/early/2025/05/12/2025.05.09.25327292},
	eprint = {https://www.medrxiv.org/content/early/2025/05/12/2025.05.09.25327292.full.pdf},
	journal = {medRxiv}
}
```

## License
This work is licensed under [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International](https://creativecommons.org/licenses/by-nc-sa/4.0/).