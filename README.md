# MICRO23 TeAAL Artifact

This repository provides the evaluation setups for the MICRO22 artifact evaluation for the paper *TeAAL: A Declarative Framework for Modeling Sparse Tensor Accelerators*. We provide a docker environment and Jupyter notebook for the artifact evaluation.

## System Requirements

### Hardware Requirements

- Memory: 256 GB
- Storage: 75 GB

### Software Requirements

- Please install [docker app](https://www.docker.com/products/docker-desktop/)

## Perform the Artifact Evaluation

### Step 0: Clone the repository

```bash
git clone https://github.com/FPSG-UIUC/micro23-teaal-artifact.git
cd micro23-teaal-artifact
```

### Step 1: Prepare your `docker-compose.yaml`

Copy the `docker-compose.yaml.template` file to a new `docker-compose.yaml`.

```bash
cp docker-compose.yaml.template docker-compose.yaml
```

Add the appropriate `USER_UID` and `USER_GID`.

### Step 2: Pull the docker image

We provide two options for obtaining the docker image. Please choose one of the options listed below.

#### Option 1: Use `docker-compose`

```bash
docker-compose pull
```

If this does nothing, proceed to Option 2.

#### Option 2: Use `docker`

```bash
docker pull timeloopaccelergy/timeloop-accelergy-pytorch:teaal-amd64
```

### Step 3: Start the container

```bash
docker-compose up -d
```

Navigate to `localhost:8888` (or `<remote_ip>:8888` if using a remote machine) within a web browser.

### Step 4: Run the experiments

Navigate to `notebooks/figs8and9.ipynb` and follow the instructions. The heading describe the graphs to compare.

### Troubleshooting

#### Missing import

To install a missing import, open the terminal within the Jupyter Lab and execute the following.

```bash
pip install <dependency>
```

`fibertree` can be installed with:

```bash
pip install git+https://github.com/FPSG-UIUC/fibertree.git@metrics#egg=fiber-tree
```

`teaal` can be installed with:

```bash
pip install git+https://github.com/FPSG-UIUC/teaal-compiler.git@modeling#egg=teaal
```

Once an import is installed, you must restart the notebook's kernel to make it visible.

You can confirm that all imports are available by running a simple copy kernel at `notebooks/simple-teaal.ipynb`.

#### Corrupted specifications

To confirm that all accelerator specifications are configured correctly, please execute `notebooks/simple-accel.ipynb`.

If any accelerator tests return `False`, restore all specifications.

```bash
cd path/to/micro23-teaal-artifact/
git restore yamls/
```

#### Not enough hardware resources

The described hardware resouces are *required* for these experiments. However, you can still generate the graphs using pre-generated simulation data.

To do so, navigate to `notebooks/figs8and9.ipynb` and after running the second Code cell, set the `pregenerated` widget to `True`. Execute the rest of the notebook, skipping cells beginning with `%%python3 -O`.

## Acknowledgements

This README is based on the instructions for the [MICRO22 Sparseloop Artifact](https://github.com/Accelergy-Project/micro22-sparseloop-artifact/).
