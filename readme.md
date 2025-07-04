# synthnn

A Python implementation of the Synthetic Nearest Neighbors (SNN) estimator for causal inference with panel data. The SNN estimator imputes each treated observation's untreated potential outcome using a synthetic nearest-neighbor "donor" pool, then averages the resulting effects to obtain the Average Treatment Effect on the Treated (ATT).

## Features

- **Flexible Panel Data Analysis**: Supports both simultaneous and staggered treatment adoption
- **Multiple Inference Methods**: Jackknife, bootstrap, or Fisher-style placebo tests for uncertainty quantification
- **Rich Visualization**: Built-in plotting for gap plots and counterfactual comparisons
- **Customizable Imputation**: Configurable SNN parameters for different data characteristics
 
## Installation

You can install `synthnn` from PyPI using pip:

```bash
pip install synthnn
```

## Quick Start

```python
import pandas as pd
from synthnn import SNN

# Load your panel data
df = pd.read_csv("your_panel_data.csv")

# Initialize the SNN model
model = SNN(
    unit_col="Unit",
    time_col="Time", 
    outcome_col="Y",
    treat_col="W",
    variance_type="bootstrap",
    resamples=500,
    alpha=0.05
)

# Fit the model and get results
model.fit(df)
model.summary()

# Generate visualizations
model.plot("gap")  # ATT over time
model.plot("counterfactual")  # Observed vs counterfactual paths
```

## Complete Example: Replicating Abadie et al. (2010)

This example demonstrates how to use SNN to replicate the famous California tobacco study. The prop99.csv file can be found in the tests folder of this repository.

```python
import pandas as pd
from synthnn import SNN

# -------------------------------------------------
# 1. Load the data from Abadie et al. (2010)
# -------------------------------------------------
df0 = pd.read_csv("prop99.csv", low_memory=False)

df = (
    df0
    .query("TopicDesc == 'The Tax Burden on Tobacco' "
           "and SubMeasureDesc == 'Cigarette Consumption (Pack Sales Per Capita)'")
    .loc[:, ["LocationDesc", "Year", "Data_Value"]]
    .rename(columns={
        "LocationDesc": "Unit",
        "Year":         "Time",
        "Data_Value":   "Y"
    })
)

# Drop territories & aggregate rows (keep 50 states + DC)
bad_units = ["District of Columbia", "United States", "Guam",
             "Puerto Rico", "American Samoa", "Virgin Islands"]
df = df[~df["Unit"].isin(bad_units)]

# ---------------------------------
# 2. Define the treatment indicator
# ---------------------------------
df["W"] = ((df["Unit"] == "California") & (df["Time"] >= 1989)).astype(int)

# ---------------------------------
# 3. Fit Synthetic-Nearest-Neighbors
# ---------------------------------
model = SNN(
    unit_col="Unit",
    time_col="Time",
    outcome_col="Y",
    treat_col="W",
    variance_type="bootstrap",
    resamples=100,
    alpha=0.05
)

model.fit(df)

# 4. Inspect results
model.summary()

# 5. Plot the gap between treated and counterfactual
model.plot(
    title="SNN replication of Abadie et al. (2010)",
    xlabel="Event Time (0 = 1989)",
    ylabel="ATT (packs per-capita)"
)

# 6. Plot observed vs counterfactual paths
model.plot(
    plot_type="counterfactual",
    title="Observed vs Synthetic California",
    xlabel="Event Time (0 = 1989)",
    ylabel="Cigarette Consumption (packs per-capita)"
)

# 7. Same as before but with calendar time on the x-axis, only post-treatment periods, and custom colors
model.plot(
    plot_type="counterfactual",
    calendar_time=True,
    xrange=(1989, 2014),
    title="Observed vs Synthetic California: Post-Treatment Periods",
    xlabel="Year",
    ylabel="Cigarette Consumption (packs per-capita)",
    counterfactual_color="#406B34",  # green
    observed_color="#ff7f0e"         # orange
)
```
### Ouput
After running the above code, you will see the following output:
```plaintext
============================================================
SNN Estimation Results
============================================================

--- Overall ATT ---
estimate    method    se p_value ci_lower ci_upper
  -28.25 bootstrap 2.236       0   -31.26   -23.95


--- ATT by Event Time (Post-Treatment) ---

event_time    att N_units    se   p_value ci_lower ci_upper    method
         0  -14.2       1  1.58         0   -17.35   -11.78 bootstrap
         1 -15.15       1 2.144 1.583e-12   -18.66   -11.68 bootstrap
         2 -22.02       1 2.186         0    -25.8   -18.67 bootstrap
         3 -22.12       1 2.336         0   -26.24   -18.12 bootstrap
         4 -25.27       1 2.196         0   -29.02   -21.48 bootstrap
         5 -29.18       1 2.288         0   -33.18   -25.17 bootstrap
         6 -31.54       1  2.45         0      -35   -27.16 bootstrap
         7 -31.75       1 2.451         0   -35.18   -26.85 bootstrap
         8 -32.37       1 2.603         0   -36.18   -27.17 bootstrap
         9  -32.8       1  2.43         0   -36.24   -27.86 bootstrap
        10 -35.09       1 2.314         0   -38.65   -30.43 bootstrap
        11 -35.74       1 2.362         0   -39.23   -30.81 bootstrap
        12 -36.65       1 2.433         0   -39.78   -31.74 bootstrap
        13 -37.07       1 2.324         0    -40.4   -32.44 bootstrap
        14 -37.75       1 3.142         0   -42.31   -31.82 bootstrap
        15 -34.89       1 3.086         0   -38.97   -28.75 bootstrap
        16 -33.71       1 3.339         0   -38.62    -27.3 bootstrap
        17  -31.7       1 3.159         0   -36.25   -25.85 bootstrap
        18 -30.94       1 3.362         0   -36.37    -24.6 bootstrap
        19 -27.91       1  2.82         0   -32.58   -22.54 bootstrap
        20 -26.63       1  2.67         0   -31.17   -21.61 bootstrap
        21 -23.79       1 2.394         0   -27.69   -19.27 bootstrap
        22 -22.49       1 2.255         0   -26.36   -18.47 bootstrap
        23 -21.83       1 2.193         0   -25.66   -17.93 bootstrap
        24 -21.35       1 2.121         0   -25.51   -17.55 bootstrap
        25 -20.63       1 1.973         0   -24.46    -17.2 bootstrap

============================================================
```
### Visualizations
![](https://github.com/rivkalipko/synthnn/blob/main/tests/gap.png?raw=true)
![](https://github.com/rivkalipko/synthnn/blob/main/tests/counterfactual.png?raw=true)
![](https://github.com/rivkalipko/synthnn/blob/main/tests/graphics.png?raw=true)

## Parameters

### SNN Class Parameters

- **unit_col, time_col, outcome_col, treat_col** (str): Column names for unit ID, time, outcome, and treatment indicator
- **variance_type** (str): Method for uncertainty quantification:
  - `"jackknife"`: Leave-one-unit-out resampling
  - `"bootstrap"`: Block bootstrap on units (default)
  - `"placebo"`: Fisher randomization test
- **resamples** (int): Number of bootstrap/placebo resamples (default: 500)
- **alpha** (float): Significance level for confidence intervals (default: 0.05)
- **snn_params** (dict): Parameters for the underlying SyntheticNearestNeighbors imputer

### SNN Imputation Parameters

The `snn_params` dictionary can include:

- **n_neighbors** (int): Number of nearest neighbors to use (default: 1)
- **weights** (str): Weight function for neighbors (`'uniform'` or `'distance'`)
- **random_splits** (bool): Whether to use random splits in the algorithm
- **max_rank** (int): Maximum rank for low-rank approximations
- **spectral_t** (float): Spectral threshold parameter (default: 0.1)
- **linear_span_eps** (float): Linear span epsilon (default: 0.1)
- **subspace_eps** (float): Subspace epsilon (default: 0.1)
- **min_value, max_value** (float): Bounds for imputed values
- **verbose** (bool): Whether to print progress information

### Plot Parameters

- **plot_type** (str): `"gap"` for ATT series or `"counterfactual"` for observed vs counterfactual paths
- **calendar_time** (bool): Use calendar time instead of event time (simultaneous adoption only)
- **xrange** (tuple): Range for x-axis as `(min, max)`
- **title, xlabel, ylabel** (str): Custom labels
- **figsize** (tuple): Figure size as `(width, height)`
- **color, observed_color, counterfactual_color** (str): Custom colors for different plot elements

## Output

After fitting, the model provides several key attributes:

- **overall_att_**: Overall average treatment effect with inference statistics
- **att_by_event_time_**: ATT series by event time with confidence intervals
- **att_by_time_**: ATT series by calendar time
- **individual_effects_**: Unit-level treatment effects
- **counterfactual_event_df_**: Observed vs counterfactual paths by event time
- **counterfactual_df_**: Observed vs counterfactual paths by calendar time

## Requirements

- pandas
- numpy
- scipy
- plotly
- kaleido for exporting images
- The `SyntheticNearestNeighbors` base class (included in package)

## Acknowledgments

The implementation in this package adapts and builds upon the code from the [`syntheticNN`](https://github.com/deshen24/syntheticNN) repository by Dennis Shen.

## License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/rivkalipko/synthnn/blob/main/LICENSE) file for details.

## Citation

If you use this package in your research, you can cite it as below.

```
@software{synthnn,
  author = {Lipkovitz, Rivka},
  month = jun,
  title = {{synthnn: a Python package for estimating treatment effects using Synthetic Nearest Neighbors}},
  url = {[https://github.com/rivkalipko/synthnn](https://github.com/rivkalipko/synthnn)},
  year = {2025}
}
```

Please also consider citing the authors of the original paper:

> Agarwal, A., Dahleh, M., Shah, D., & Shen, D. (2023, July). Causal matrix completion. In *The thirty sixth annual conference on learning theory* (pp. 3821-3826). PMLR.
