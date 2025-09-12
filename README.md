This code generates the dataset used in the paper Scaling and Population Loss in Mexican Urban Centres, to be published.

To generate the data use `uv` to install the required Python version and dependencies specified in `pyproject.toml`.
One the enviroment is available, the following `snakemake` commands can be used to generate the data set:

```
uv run snakemake gen_rfuncs --cores 1
```

This generates the output files `mesh.geoparquet` (the multi-temporal population grid), and the radial density functions as csv files in `outputs/rafial_f/`.

