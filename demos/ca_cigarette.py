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