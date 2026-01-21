# STAF Deep Neural Network Potential

This repository contains scripts to train the **STAF Deep Neural Network Potential (DNNP)** introduced in:

> DOI: **10.1063/5.0139245**

The code also includes an extension to **atomic mixtures**. The framework is designed to train neural-network-based interatomic potentials and to evaluate energy and force predictions.

---

## Features

* Training of deep neural network interatomic potentials
* Support for single and double precision (GPU)
* Extension to atomic mixtures
* Energy and force inference
* Dataset generation from raw molecular dynamics data

---

## Installation

### Requirements

The following Python packages are required:

* **TensorFlow** (GPU version recommended)
* **PyYAML**

> ⚠️ Note: the original implementation was developed with TensorFlow 2.8. Newer versions may work but are not fully guaranteed.

### Recommended setup (Conda)

It is recommended to create a dedicated Conda environment:

```bash
conda create -n staf_nnp python=3.9
conda activate staf_nnp
conda install tensorflow-gpu==2.14
conda install pyyaml
```

### CUDA compilation

The folders:

* `AlphaNesGpu_float`
* `AlphaNesGpu_double`

contain the GPU implementations for **single** and **double precision**, respectively.

Inside each folder you will find:

* `install_path.sh` (Linux)


Edit the script to correctly set:

* g++ compiler path
* CUDA compiler path
* CUDA libraries path
* Python path

Then run:

```bash
bash install_path.sh
```

A successful installation will compile the `*.cu.cc` source files located in the `src/` directory.

---

## Dataset Preparation

The training dataset must include:

* Atomic trajectories
* Simulation box information
* Energies
* Forces

To generate the dataset from raw data, use:

```bash
python make_dataset_from_raw.py
```

This script searchs pos.dat, force.dat, box.dat and energy.dat files organized with one row for frames and it takes a seed as the only command line input for the split process of the dataset. Finally it creates the training and test datasets in the required internal format.

---

## Input Configuration Files

The training and inference processes are fully controlled through YAML configuration files.

### Training input (`input.yaml`)

Typical parameters include:

* **model**

  * network architecture
  * activation functions
  * precision (float / double)

* **training**

  * number of epochs
  * batch size
  * learning rate
  * optimizer settings

* **data**

  * path to dataset
  * training / validation split

* **loss**

  * energy weight
  * force weight

> 📌 Refer to the example `input.yaml` provided in the repository for the full list of available parameters.

### Inference input (`input_inference.yaml`)

The inference configuration defines:

* trained model path
* dataset for evaluation
* output options for energy and force errors

---

## Running a Training Example

Once the environment is configured and the dataset is prepared, a training can be started with:

```bash
python alpha_nnpes_full_main.py input.yaml
```

During training, the code will:

* read the dataset
* train the neural network potential
* save checkpoints and final model parameters

---

## Running Inference

After training, energy and force errors can be computed using:

```bash
python alpha_nnpes_full_inference_main.py input_inference.yaml
```

The results include quantitative error metrics on the test dataset.

---

## Development Models

The `DEV/` folder contains experimental and under-development models, including:

* Neural Network Coarse-Graining (NN-CG) model (to be presented in a forthcoming publication)
* A model for training directly from radial distribution functions

These components are **experimental** and subject to change.

---

## Citation

If you use this code in your research, please cite:

```
Author(s), Journal, Year
DOI: 10.1063/5.0139245
```

---

## License

Specify the license here (e.g., MIT, GPL, BSD).

---

## Contact

For questions or collaborations, please open an issue or contact the authors directly.


