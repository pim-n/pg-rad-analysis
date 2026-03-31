# pg-rad-analysis
Scripts and notebooks for analysis of PG-RAD.

See the [repo of PG-RAD](https://github.com/pim-n/pg-rad).

## Install

With [miniforge](https://github.com/conda-forge/miniforge) installed, run the following:

```
git clone https://github.com/pim-n/pg-rad-analysis
cd pg-rad-analysis
conda env create -f environment.yml
```

with the conda environment activated, you may try to run

```
pgrad --example --showplots
```

which should produce a test landscape with two sources and show the plots.
