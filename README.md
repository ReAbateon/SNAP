# Speculative Node Anticipation Process (SNAP)
> Tool implementing the SNAP approach to improve the online execution of DT based models.

> **Note:** This repository was created to support the review of *Advancing Decision Tree-based Online Inference for Edge Computing*, but it is openly available and usable by the community.

## Overview
SNAP (Speculative Node Anticipation Process) is a custom vectorized kernel for accelerating decision-tree (DT) inference on edge devices. The approach leverages SIMD vector extensions to speed up tree traversal by *speculatively* evaluating candidate root-to-leaf paths.

Instead of resolving one branch decision at a time, SNAP hypothesizes a traversal path and validates, in parallel, all split predicates along that path within a single vectorized step. If the speculation is correct, the inference completes immediately. Otherwise, SNAP identifies the first mismatching node and redirects the traversal by selecting an alternative path within the corresponding sub-tree, repeating the process until a valid path is found.

This repository provides the tools required to reproduce the experiments. Starting from a joblib file containing a single decision-tree model, the tool automatically generates a C header file that includes all the data structures needed to execute the SNAP technique.

## Project Structure
The repository is organized as follows:

```text
├── C_files/                # C source files
│   ├── Kernel.c            # SNAP vectorized kernel
│   ├── example.c           # Example usage
├── Synthetic_Dataset/      # Synthetic dataset used in the experiments
│   ├── synthetic_timeseries_test.csv
│   ├── synthetic_timeseries_train.csv
├── Tool/                   # Model training and code generation tools
│   ├── CodeGeneratorDT.py  # Generates SNAP C headers
│   ├── trainerDT.py        # Trains single DT models and exports joblib
├── README.md
└── LICENSE
```

The `C_files` directory contains the file `example.c`, which provides a ready-to-use main program demonstrating how to run the SNAP kernel, as well as the C implementation of the kernel itself in `Kernel.c`.

The `Synthetic_Dataset` directory contains the CSV files with the training and test samples of the artificially generated dataset.

The `Tool` directory includes the `CodeGeneratorDT.py` script, which generates the SNAP-compatible C header code to be imported into an embedded project using ARM Helium. The `trainerDT.py` script trains a single decision-tree model starting from a training and a test set and exports a compatible joblib file.

## Synthetic Timeseries Dataset
This repository provides a small **synthetic multivariate time-series dataset** designed for evaluating machine learning models for **multi-class anomaly detection** in industrial-like sensor streams.

The dataset emulates signals acquired from a monitoring system where samples are collected **sequentially in time**, exhibiting strong temporal correlation, smooth dynamics, and realistic noise patterns.

It features:
- 5000 training samples and 2000 test samples
- Multivariate time-series
- 12 sensor-like features
- 4 classes (normal + 3 anomaly types)

This is a **synthetic dataset** and does not represent any specific real-world system.  
It is provided for research, prototyping, and benchmarking purposes only.

## Requirements

This section outlines the requirements needed to use the python scripts and to integrate the generated code.

### Python Requirements
To run the code generation tool, the following requirements must be satisfied:

- **Python version:** Python 3.13.5  
- A standard Python environment capable of running scientific and machine learning libraries.

The tool has been tested on macOS Tahoe 26.2.

The following external libraries are required to run the tool:

- `numpy`
- `pandas`
- `scikit-learn`
- `joblib`

All required external dependencies can be installed using `pip`:

```bash
pip install numpy pandas scikit-learn joblib
```

### Generated Code Requirements

The generated code is provided as a C header file and can be included in any embedded C project.

To correctly compile and execute the generated code, the following requirements must be satisfied:

- A target platform based on an **ARM processor**.
- Support for **ARM Helium vector extensions**, as the code relies on a SIMD-based execution model.
- Compilation using an **ARM bare-metal toolchain**, such as `arm-none-eabi-gcc`.

## Usage

### 1) Train a Decision Tree model

Run the following command to train a single decision-tree model and generate a compatible joblib file:

```bash
python3 Tool/trainerDT.py \
  --train Synthetic_Dataset/synthetic_timeseries_train.csv \
  --test Synthetic_Dataset/synthetic_timeseries_test.csv \
  --max-depth 8 \
  --seed 0 \
  --weight 8
```
### 2) Generate SNAP C header

Use the generated joblib file to produce the SNAP-compatible C header:
```bash
python3 Tool/CodeGeneratorDT.py \
  example.joblib \
  -V 8 \
  --testpath example_test.csv
```

### 3) Generated Code Usage

The generated C code must be imported into the target embedded project and included in the main application source file.  
All required headers and dependencies needed for execution are automatically included by the generated code.

To correctly place the generated data structures, it is necessary to initialize the appropriate memory regions in the linker script.  
An example linker script configuration is provided in the repository and can be used as a reference:

> **Linker script example:** `C_files/linker_script_example.ld`

#### Available Functions

The generated code exposes the following functions, which can be called directly from the application:

- `init()`  
  Initializes the starting paths.

- `inference(int16_t* sample)`  
  Executes the SNAP kernel and returns the predicted class label for the input sample.

- The predicted output label is stored in the global variable `out`.

## License
This project is licensed under the **GNU General Public License v3.0 (GPLv3)**.

For more details, see the `LICENSE` file included in this repository.
  

