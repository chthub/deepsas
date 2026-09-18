# Run DeepSAS on subsampled AnnData objects

This workflow creates subsamples, runs DeepSAS for each one, and generates the
summary tables for every completed experiment.


### Usage

1. Create the subsamples:

```bash
uv run python subsampling.py --input your_data.h5ad --subsample_size 30000 --output ./subsamples
```
2. In a GPU compute allocation, run DeepSAS on every subsample. Do not run the
   batch loop with `nohup` on an OSC login node.

```bash
bash run_experiments.sh ./subsamples ./outputs 0 Data
```

The arguments are `SUBSAMPLE_DIR`, optional `OUTPUT_DIR`, optional CUDA device
index, and optional experiment-name prefix.

3. After all training runs finish successfully, generate their tables using the
   same input directory, output directory, and experiment prefix:

```bash
bash Generate_table_sampling.sh ./subsamples ./outputs Data
```

Training saves the exact processed AnnData object inside each experiment
directory. Table generation reads that saved object and the final result, so it
does not need the original `--input_data_count` again.

For the visualization and downstream analysis of SnCs and SnGs, please follow the tutorial in the [`tutorial.ipynb`](./tutorial.ipynb).


#### Important arguments

- `--output_dir`: Directory to store output files and results.
- `--exp_name`: Descriptive name for the experiment.
- `--device_index`: GPU device index if CUDA is available.
