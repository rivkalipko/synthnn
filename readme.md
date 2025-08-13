# synthnn

A Python package for panel data causal inference implementing synthetic nearest neighbors (SNN), a causal model for matrix completion that imputes treated units’ counterfactual outcomes from weighted nearest neighbors in a low-rank subspace learned from pre-treatment data..

## Features

* **Flexible Panel Data Support** — Works with both simultaneous and staggered treatment adoption.
* **Multiple Inference Methods** — Jackknife, bootstrap, and Fisher-style placebo tests for uncertainty quantification.
* **Built-in Visualization** — Gap plots and observed vs. counterfactual comparisons.
* **Customizable Imputation** — Fully configurable parameters to match your data’s characteristics.

## Installation

```bash
pip install synthnn
```

## Quick Start

```python
import pandas as pd
from synthnn import SNN

# Load your panel data
df = pd.read_csv("your_panel_data.csv")

# Initialize and fit the SNN model
model = SNN(
    unit_col="Unit",
    time_col="Time", 
    outcome_col="Y",
    treat_col="W",
    variance_type="bootstrap",
    resamples=500,
    alpha=0.05
)
model.fit(df)
model.summary()

# Visualize results
model.plot("gap")              # Average treatment effect on the treated (ATT) over time
model.plot("counterfactual")   # Observed vs. counterfactual
```

## Full Example — Replicating Abadie et al. (2010)

This example reproduces the well-known California tobacco control study.
Data: [`prop99.csv`](https://github.com/rivkalipko/synthnn/blob/main/demos/prop99.csv) in the `demos` folder.

```python
import pandas as pd
from synthnn import SNN

# 1. Load the data from Abadie et al. (2010)
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

# Drop territories & aggregate rows (keep 50 states)
bad_units = ["District of Columbia", "United States", "Guam",
             "Puerto Rico", "American Samoa", "Virgin Islands"]
df = df[~df["Unit"].isin(bad_units)]

# 2. Define the treatment indicator
df["W"] = ((df["Unit"] == "California") & (df["Time"] >= 1989)).astype(int)

# 3. Fit Synthetic-Nearest-Neighbors
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
).write_image("gap.png")

# 6. Plot observed vs counterfactual paths
model.plot(
    plot_type="counterfactual",
    title="Observed vs Synthetic California",
    xlabel="Event Time (0 = 1989)",
    ylabel="Cigarette Consumption (packs per-capita)"
).write_image("counterfactual.png")

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
).write_image("graphics.png")

# 8. Inference using the placebo test (only works if there is exactly one treated unit)
model_pc = SNN(unit_col="Unit", time_col="Time", outcome_col="Y", treat_col="W",
               variance_type="placebo", alpha=0.05)
model_pc.fit(df)
model_pc.summary()

# 9. Plot the results, displaying the paths of the placebo treated units against the actual treated unit
model_pc.plot(show_placebos=True,
              title="Placebo Test for Inference",
              xlabel="Event Time (0 = 1989)",
              ylabel="ATT (packs per capita)").write_image("placebo.png")
```
### Output
<details>
<summary>Click to expand</summary>

```plaintext
============================================================
SNN Estimation Results
============================================================

--- Overall ATT ---
estimate    method    se p_value ci_lower ci_upper
  -28.25 bootstrap 2.032       0   -32.07   -24.03


--- ATT by Event Time (Post-Treatment) ---

event_time    att N_units    se   p_value ci_lower ci_upper    method
         0  -14.2       1 1.651         0   -17.06   -11.28 bootstrap
         1 -15.15       1 2.077 3.015e-13   -18.75   -11.43 bootstrap
         2 -22.02       1 2.089         0   -26.16   -18.22 bootstrap
         3 -22.12       1 2.184         0   -26.15   -18.05 bootstrap
         4 -25.27       1 1.959         0   -28.55   -21.33 bootstrap
         5 -29.18       1 2.129         0   -32.97      -25 bootstrap
         6 -31.54       1 2.052         0   -35.08    -27.1 bootstrap
         7 -31.75       1 2.054         0    -35.6   -27.29 bootstrap
         8 -32.37       1 2.207         0    -36.2   -28.41 bootstrap
         9  -32.8       1 2.035         0   -36.08   -28.68 bootstrap
        10 -35.09       1 2.144         0   -38.64   -31.03 bootstrap
        11 -35.74       1 2.196         0   -39.74   -31.06 bootstrap
        12 -36.65       1 2.301         0   -41.26   -31.28 bootstrap
        13 -37.07       1 2.291         0    -41.5   -31.68 bootstrap
        14 -37.75       1 3.217         0   -44.07   -31.11 bootstrap
        15 -34.89       1 3.052         0   -40.54   -27.46 bootstrap
        16 -33.71       1 3.303         0   -39.55   -26.32 bootstrap
        17  -31.7       1 3.097         0   -37.31   -25.12 bootstrap
        18 -30.94       1 3.264         0    -36.9   -23.89 bootstrap
        19 -27.91       1 2.687         0   -32.99   -22.78 bootstrap
        20 -26.63       1 2.583         0   -31.33   -21.51 bootstrap
        21 -23.79       1 2.254         0   -27.74   -19.66 bootstrap
        22 -22.49       1 2.131         0   -26.36   -18.57 bootstrap
        23 -21.83       1 2.042         0   -25.58   -18.39 bootstrap
        24 -21.35       1 2.044         0   -24.94   -17.73 bootstrap
        25 -20.63       1 1.895         0   -24.19   -17.52 bootstrap

============================================================
============================================================
SNN Estimation Results
============================================================

--- Overall ATT ---
estimate placebo_p placebo_rank
  -28.25      0.08            4

Placebo Fisher p-value: 0.08  (rank 4/50)


--- ATT by Event Time (Post-Treatment) ---

 event_time    att N_units placebo_p
          0  -14.2       1       0.2
          1 -15.15       1      0.22
          2 -22.02       1      0.12
          3 -22.12       1      0.12
          4 -25.27       1      0.08
          5 -29.18       1      0.06
          6 -31.54       1      0.06
          7 -31.75       1      0.06
          8 -32.37       1      0.06
          9  -32.8       1      0.04
         10 -35.09       1      0.04
         11 -35.74       1      0.04
         12 -36.65       1      0.04
         13 -37.07       1      0.06
         14 -37.75       1       0.1
         15 -34.89       1      0.12
         16 -33.71       1       0.1
         17  -31.7       1      0.14
         18 -30.94       1      0.14
         19 -27.91       1      0.14
         20 -26.63       1       0.2
         21 -23.79       1       0.2
         22 -22.49       1      0.18
         23 -21.83       1      0.18
         24 -21.35       1      0.16
         25 -20.63       1      0.12

============================================================
```

</details>

### Plots

![](https://github.com/rivkalipko/synthnn/blob/main/demos/gap.png?raw=true)
![](https://github.com/rivkalipko/synthnn/blob/main/demos/counterfactual.png?raw=true)
![](https://github.com/rivkalipko/synthnn/blob/main/demos/graphics.png?raw=true)
![](https://github.com/rivkalipko/synthnn/blob/main/demos/placebo.png?raw=true)

## Parameters

### General

* **unit\_col, time\_col, outcome\_col, treat\_col** *(str)* — Column names for unit ID, time, outcome, and treatment indicator.
* **variance\_type** *(str)* — Inference method:

  * `"jackknife"` — Leave-one-unit-out resampling
  * `"bootstrap"` *(default)* — Block bootstrap on units
  * `"placebo"` — Fisher randomization test (only when exactly one treated unit)
* **resamples** *(int)* — Bootstrap resamples (default: 500)
* **alpha** *(float)* — Significance level for confidence intervals (default: 0.05)
* **snn\_params** *(dict)* — Parameters for the `SyntheticNearestNeighbors` imputer.

### SNN Parameters (`snn_params`)

* **n\_neighbors** *(int)* — Number of nearest neighbors (default: 1)
* **weights** *(str)* — `'uniform'` or `'distance'`
* **random\_splits** *(bool)* — Use random splits in the algorithm
* **max\_rank** *(int)* — Maximum rank for low-rank approximation
* **spectral\_t, linear\_span\_eps, subspace\_eps** *(float)* — Algorithm thresholds (default: 0.1)
* **min\_value, max\_value** *(float)* — Bounds for imputed values
* **verbose** *(bool)* — Print progress.

### Plot Parameters

* **plot\_type** — `"gap"` or `"counterfactual"`
* **calendar\_time** *(bool)* — Use calendar time (for simultaneous adoption only)
* **xrange** *(tuple)* — `(min, max)` for x-axis
* **title, xlabel, ylabel** *(str)* — Labels
* **figsize** *(tuple)* — `(width, height)`
* **color, observed\_color, counterfactual\_color, placebo\_color** *(str)* — Plot colors
* **placebo\_opacity** *(float)* — Opacity for placebo lines (default: 0.25)

## Output Attributes

After fitting, the model exposes:

* **overall\_att\_** — Overall ATT with inference statistics
* **att\_by\_event\_time\_** — ATT series by event time
* **att\_by\_time\_** — ATT series by calendar time
* **individual\_effects\_** — Unit-level effects
* **counterfactual\_event\_df\_** — Observed vs. counterfactual (event time)
* **counterfactual\_df\_** — Observed vs. counterfactual (calendar time)

## Requirements

* `pandas`, `numpy`, `scipy`, `plotly`, `scikit-learn`

## Acknowledgments

The implementation in this package adapts and builds upon the code from the [syntheticNN](https://github.com/deshen24/syntheticNN) repository by Dennis Shen.

## License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/rivkalipko/synthnn/blob/main/LICENSE) file for details.

## Citation

If you use this package in your research, you can cite it as below.
```
@software{synthnn,
  author = {Lipkovitz, Rivka},
  month = jun,
  title = {{synthnn: a Python package for estimating treatment effects using Synthetic Nearest Neighbors}},
  url = {https://github.com/rivkalipko/synthnn},
  year = {2025}
}
```

Please also consider citing the authors of the original paper:

> Agarwal, A., Dahleh, M., Shah, D., & Shen, D. (2023, July). Causal matrix completion. In *The thirty sixth annual conference on learning theory* (pp. 3821-3826). PMLR.