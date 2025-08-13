import pandas as pd
import numpy as np
from src.synthnn import SNN

# --- 1. Generate Sample Panel Data ---
# Create a dataset with 10 units over 10 time periods.
# Units 0-4 are treated at different times (staggered adoption).
# Units 5-9 are never-treated controls.
np.random.seed(1)
n_units = 10
n_periods = 10
units = range(n_units)
periods = range(n_periods)

# Create base DataFrame
df = pd.DataFrame([(u, t) for u in units for t in periods], columns=['Unit', 'Time'])

# Add unit and time fixed effects
unit_effects = {u: np.random.normal(0, 2) for u in units}
time_effects = {t: t * 0.5 for t in periods}
df['Y'] = df['Unit'].map(unit_effects) + df['Time'].map(time_effects)

# Define treatment start times for treated units
treatment_start_times = {0: 5, 1: 6, 2: 6, 3: 7, 4: 8}
df['W'] = 0

# Apply treatment effect (ATT = 3)
for unit, start_time in treatment_start_times.items():
    df.loc[(df['Unit'] == unit) & (df['Time'] >= start_time), 'W'] = 1
    df.loc[(df['Unit'] == unit) & (df['Time'] >= start_time), 'Y'] += 3

# Add some noise
df['Y'] += np.random.normal(0, 0.5, len(df))

print("Sample Data Head:")
print(df.head())
print("\nTreatment Status:")
print(pd.crosstab(df['Unit'], df['W']))

# --- 2. Initialize and Fit the SNN Model ---
# We will use Bootstrap for standard errors.
model = SNN(
    unit_col="Unit",
    time_col="Time",
    outcome_col="Y",
    treat_col="W",
    resamples= 50,
)

# Fit the model to the data
model.fit(df)

# --- 3. View the Results ---
# Print a comprehensive summary of the results
model.summary()

# --- 4. Plot the Event Study ---
# Generate and display the event study plot
model.plot(
    title="SNN Event Study: ATT on Outcome Y",
)

# --- 5. Access Results Programmatically ---
# You can also access the results as DataFrames for further analysis
overall_att = model.overall_att_
event_study_results = model.att_by_event_time_

print("\nOverall ATT Estimate:")
print(overall_att)
