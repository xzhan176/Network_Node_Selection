# K Node

## Get Started

Conda is used to manage the environment for this project. The environment file is `environment.yml`.

### Create Conda environment

```bash
conda env create --file environment.yml
```

### Activate Conda environment

```bash
conda activate opinion_polarization
```

## Run the simulation

```bash
python run.py <network_name> <k_value> <experiment_index>
```

You can customize the `run.py` script parameters to run the simulation with different configurations. To view available options, run

```bash
python run.py --help
```

## Run the simulation in batch

```bash
python run_batch.py <network_name> <k_value>
```

You can customize the `run_batch.py` script parameters to run the simulation with different configurations. To view available options, run

```bash
python run_batch.py --help
```

## Results

The results of the simulation are stored in the `results` directory.
