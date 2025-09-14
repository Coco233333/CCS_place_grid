# CCS_place_grid

Content: This repository contains the code and analysis scripts for the manuscript "Localized Space Coding and Phase Coding Complement Each Other to Achieve Robust and Efficient Spatial Representation". It facilitates the full reproduction of all results, analyses, and figures reported in the study.

## 🗂️ Repository Overview

A brief overview of the project structure: \
├── Beyesian_integration/ # Analysis and visualization for Figure 3 \
├── Non_local_error/ # Analysis and visualization for Figure 4 \
├── Storing_multiple_maps/ # Analysis and visualization for Figure 5 & 6 \
├── environment.yml # Conda environment specification \
└── README.md # This file

### Beyesian_integration
Figure 3a-b \
Bayesian_integration/Bayesian_integration_distribution.ipynb \
Figure 3c-e \
Data: Bayesian_integration/Bayesian_integration.py \
Visualization: Bayesian_integration/load_data.ipynb \
Figure S3a  \
Bayesian_integration/Correlated_noise.ipynb \
Figure S3b  \
Bayesian_integration/corr_bayesian_integration.ipynb \
Figure S4  \
Bayesian_integration/Parameter_sensitivity.ipynb 

### Non_local_error
Figure 4c: example distributions of decoding errors \
Non_local_error/Net_decoding.ipynb \
Figure 4d: one-step decoding performances across different noise levels \
Non_local_error/MAP_vs_Net_noise.ipynb \
Figure 4e: decoding performance over time of path integration \
Non_local_error/MAP_vs_Net_time.ipynb \
Figure S1d-e \
Non_local_error/LSC_vs_PSC.ipynb \
Figure S5 \
Non_local_error/Comparison_Constrained_model.ipynb \

### Storing_multiple_maps
Figure 5a-b\
Storing_multiple_maps/Place_cell_multiple_maps.ipynb\
Figure 5c\
Storing_multiple_maps/test/coupled_net_plot.ipynb

Parameters \
Figure 6a; Figure S6 \
Storing_multiple_maps/test/coupled_net_plot.ipynb

#### remapping
Figure 6b-c; Figure S2 \
Data: Storing_multiple_maps/Artificial_Remapping.ipynb \
Visualization: Storing_multiple_maps/final_remapping 

## ⚙️ Installation & Setup
To replicate the computational environment required to run this code:

1.  **Clone this repository:**
    ```bash
    git clone https://github.com/Coco233333/CCS_place_grid.git
    cd CCS_place_grid
    ```

2.  **Create the Conda environment** from the `environment.yml` file:
    ```bash
    conda env create -f environment.yml
    conda activate CCS_env
    ```
    
## 📜 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details. This license allows for reuse and modification with appropriate attribution.


## 📧 Contact

For questions regarding the code and analysis, please open an issue on GitHub or contact:
-   Tianhao Chu - chutianhao@stu.pku.edu.cn
-   Yuling Wu - yulingwu@stu.pku.edu.cn
-   Si Wu - siwu@pku.edu.cn
