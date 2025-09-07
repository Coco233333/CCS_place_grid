# CCS_place_grid

Content: This repository contains the code and analysis scripts for the manuscript "Localized Space Coding and Phase Coding Complement Each Other to Achieve Robust and Efficient Spatial Representation". It facilitates the full reproduction of all results, analyses, and figures reported in the study.

## 🗂️ Repository Overview

A brief overview of the project structure: \
├── Beyesian_integration/ # Analysis and visualization for Figure 2 \
├── Non_local_error/ # Analysis and visualization for Figure 4 \
├── Storing_multiple_maps/ # Analysis and visualization for Figure 5 & 6 \
├── environment.yml # Conda environment specification \
└── README.md # This file

### Beyesian_integration
Fig2a-b \
Bayesian_integration/Bayesian_integration_distribution.ipynb \
Fig2c-e \
data: \
Bayesian_integration/Bayesian_integration_notebook.ipynb\
Bayesian_integration/data/distribution_results.npz (fig2c)\
Bayesian_integration/data/results.npz (fig2d-e)\
visualisation: \
Bayesian_integration/load_data.ipynb

### Non_local_error
Fig4c: example distributions of decoding errors \
Non_local_error/Net_decoding.ipynb\
Fig4d: one-step decoding performances across different noise levels \
Non_local_error/MAP_vs_Net_noise.ipynb\
Fig4e: decoding performance over time of path integration\
Non_local_error/MAP_vs_Net_time.ipynb

### Storing_multiple_maps
Fig5a-b\
Storing_multiple_maps/Place_cell_multiple_maps_wyl.ipynb\
Fig5c\
Storing_multiple_maps/test/coupled_net_plot.ipynb

Parameters \
Fig6a; FigS9\
Storing_multiple_maps/test/coupled_net_plot.ipynb

#### remapping
Fig6; FigS5 \
Storing_multiple_maps/final_remapping 

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
