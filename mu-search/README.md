# μSearch

## Introduction

μSearch (or musearch) is a reinforcement learning based sequence design tool for navigating protein fitness landscapes. When combined with μFormer, this framework identifies good mutations across vast protein sequence spaces, providing a robust and efficient approach for protein optimization.

---

## Environment

Follow these steps to set up the environment:

1. Clone the repository:
    ```bash
    git clone https://github.com/microsoft/Mu-Protein.git
    cd Mu-Protein/mu-search
    ```

2. Create and activate a conda environment:
    ```bash
    conda create -n musearch python=3.8 -y
    conda activate musearch
    ```

3. Install FLEXS dependencies (ignore errors if they occur):
    ```bash
    cd src/flexs
    pip install -e .
    pip install -r requirements.txt
    conda install -c bioconda viennarna -y
    pip install pyrosetta-installer
    python -c 'import pyrosetta_installer; pyrosetta_installer.install_pyrosetta()'
    cd -
    ```

4. Install μSearch dependencies:
    ```bash
    conda install -c conda-forge tape_proteins=0.5 -y
    pip install -r src/flexs/flexs/landscapes/landscape/muformer/muformer_landscape/requirements.txt
    pip install -r src/flexs/flexs/baselines/explorers/requirements.txt -i https://pypi.python.org/simple/
    cd src/flexs/flexs/baselines/explorers/stable_baselines3
    pip install --upgrade setuptools packaging
    pip install -e .
    pip install numpy==1.21.5
    pip install gym==0.22.0
    pip install torch==1.11.0+cu113 -f https://download.pytorch.org/whl/torch_stable.html
    cd -
    ```

5. Download μFormer model weights: put the following two files from **[Figshare](https://figshare.com/articles/figure/_Former_checkpoint_files_used_in_Search/30227557)** into `src/flexs/flexs/landscapes/landscape/muformer/`:
   - `ur50-pcomb-prot_pmlm_1b-3x16-ckpt-checkpoint_best.pt`
   - `muformer-l-BLAT_ECOLX_Ranganathan2015_CFX.pt`

---

## Supported Features

### Fitness landscapes
Supporting the following landscapes:
1. `rna`
2. `gfp`
3. `rosetta`
4. `aav`
5. `tf` 
6. `muformer` (μFormer)

Supporting the following methods:
1. `adalead`
2. `cbas`
3. `cmaes`
4. `dynappo`
5. `BO` 
6. `gwg` 
7. `evoplay` 
8. `musearch` (μSearch)

---

## Running experiments

To run a sequence design experiment, use the following command:

```bash
python examples/baseline.py --method {method} --landscape {landscape} --sequences_batch_size {sequences_batch_size} --model_queries_per_batch {model_queries_per_batch} --run {run}
```

### Example
Use μSearch as the sequence design method and μFormer as the fitness landscape oracle:
```bash
python examples/baseline.py --method musearch_gt --landscape muformer --sequences_batch_size 100 --model_queries_per_batch 5000 --run 1
```

---

## Acknowledgements

The μSearch codebase builds upon the following projects:
- [FLEXS](https://github.com/samsinai/FLEXS)
- [Stable Baselines3](https://github.com/DLR-RM/stable-baselines3)

---
