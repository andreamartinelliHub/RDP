# 🚀 Fast-Track Time-Series Framework

A modular, registry-driven framework designed for the rapid integration of Time-Series models and datasets. This project decouples data engineering from model architecture, allowing you to swap components via configuration without touching the core pipeline.


## Table of Contents
- [UV Installation](#uv-installation-)


## 🛠️ Workflow of main.py
1. Prepare all configs with hydra and Omegaconf
2. Load data using the selected dataset to be used via DATASETS_REGISTRY
3. Adapt model_configs and load the model via MODELS_REGISTRY
4. Prepare the experiments/exp/ folder  
5. Split data and Train the model
6. Save things

## 🌿 Environment Creation
The creation of the environment is handled by uv: https://docs.astral.sh/uv/getting-started/

After cloning the repo, you can find a git-tracked file called 'requirements.yml'.
`pyproject.toml` contains all needed pip packages:
- `$ uv venv`
- `$ uv init`
- `$ uv sync`

**Updating the env**: *uv add*, *uv remove*

**Activating the env**: *source .venv/bin/activate*

## 📁 Root & SubFolders
Structure of the root folder of the project: 
<pre lang="markdown"> <code> RDP/
    ├── experiments/
    |    └── exp_name_i/ # =${time_model_dataset}
    |         ├── final_config.yaml
    |         ├── train.log
    |         ├── loss.csv
    |         └── weights/
    |              ├── first.ckpt/
    |              ├── best.ckpt/
    |              └── last.ckpt/
    ├── src/RDP/
    |    ├── __init__.py (Easy imports)
    |    ├── configs/
    |    |    ├── config_user1.yaml
    |    |    └── config_user2.yaml
    |    ├── datasets/
    |    |    ├── __init__.py (Triggering DATASETS_REGISTRY)
    |    |    ├── _add_dataset.py (Script with template for new datasets)
    |    |    └── dataset_i/
    |    |         ├── (dataset_i raw files)
    |    |         └── dataset_i.py # from raw to pd.DataFrame
    |    ├── models/
    |    |    ├── __init__.py (Triggering MODELS_REGISTRY)
    |    |    ├── _add_model.py (Script with template for new model)
    |    |    └── model_i/
    |    |         ├── (dataset_i raw files)
    |    |         └── model_i.py # init and forward logics
    |    ├── data_structure/
    |    |    └── data_structure.py (*TimeSeries* class)
    |    |    └── utils.py/
    |    ├── config.yaml
    |    ├── registry.py (*Registry* class and its instances)
    |    ├── template.py
    |    └── utils.py
    ├── main.py
    ├── pyproject.toml
    └── README.md (you are here)</code> </pre>


## 🚀 Registry Design Pattern

Class Registry with methods:
- show_all(cls): print all registry inited
- register(self, name): decorator to register class implemented in the corresponding registry instance
- get(self, name): init the model via name (the same used to register it)
