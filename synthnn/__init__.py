import pandas as pd
import numpy as np
import scipy.stats
from .base import SyntheticNearestNeighbors
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class SNN:
    r"""
    Synthetic Nearest Neighbors (SNN) estimator for panel-data causal analysis.

    The estimator imputes each treated observation’s untreated potential outcome
    using a synthetic nearest-neighbor donor pool, then averages the resulting
    effects to obtain the Average Treatment Effect on the Treated (ATT). It also
    supports jackknife, bootstrap, or Fisher-style placebo resampling for
    uncertainty quantification and provides diagnostic plots.

    ----------
    Quick start
    ----------
    >>> model = SNN(unit_col="id", time_col="year", outcome_col="y", treat_col="d",
    ...             variance_type="bootstrap", resamples=999)
    >>> model.fit(panel_df)
    >>> model.plot("gap")                 # event-time ATT (the “gap” plot)
    >>> model.plot("counterfactual")      # observed vs. counterfactual (event time)
    >>> model.summary()

    ----------
    Parameters
    ----------
    unit_col, time_col, outcome_col, treat_col : str, default "Unit", "Time", "Y", "W"
        Column names identifying unit, period, outcome, and treatment.
    snn_params : dict, optional
        Additional keyword arguments forwarded to the underlying
        :class:`SyntheticNearestNeighbors` imputer (e.g., ``{'n_neighbors': 5, 'weights': 'uniform'}``).
    variance_type : {"jackknife", "bootstrap", "placebo"}, default "bootstrap"
        Method for uncertainty quantification.

        - ``"jackknife"``: Leave-one-unit-out resampling.
        - ``"bootstrap"``: Block bootstrap on units (with replacement).
        - ``"placebo"``: Fisher-style deterministic placebo test that **requires exactly
          one treated unit**. It simulates the observed treatment start time on each
          originally untreated unit (one at a time) to form the placebo distribution.
    resamples : int, default 500
        Number of resamples used when ``variance_type="bootstrap"``.
    alpha : float, default 0.05
        Significance level for confidence intervals and two-sided p-values
        (used for ``"jackknife"`` and ``"bootstrap"``).

    ----------
    Attributes (populated after :meth:`fit`)
    ----------
    results_ : dict
        Raw store of all estimation outputs, including point estimates, standard errors,
        confidence intervals, and p-values for overall and by-event-time effects.
    overall_att_ : pandas.DataFrame
        Single-row DataFrame with point estimate and inference statistics for the overall ATT.
    att_by_time_ : pandas.DataFrame
        ATT series aggregated by calendar time. (Inference statistics are not computed for this series.)
    att_by_event_time_ : pandas.DataFrame
        ATT series aggregated by event time (relative to first treatment), with inference statistics.
    individual_effects_ : pandas.DataFrame
        Actual outcome, imputed counterfactual, and estimated effect for every treated unit-time observation.
    counterfactual_df_ : pandas.DataFrame
        Mean observed and counterfactual outcome paths for treated units, aggregated by calendar time.
    counterfactual_event_df_ : pandas.DataFrame
        Mean observed and counterfactual outcome paths for treated units, aggregated by event time.
    placebo_dist_ : numpy.ndarray
        If ``variance_type="placebo"``, distribution of placebo ATTs from the randomization test.
    placebo_event_dist_ : list[pandas.Series]
        If ``variance_type="placebo"``, list of placebo ATT paths over event time.

    ----------
    Notes
    ----------
    - No treatment reversals are assumed; periods after first treatment are coded ``1`` permanently.
    - In the printed summary, event-time output is truncated to non-negative periods (``τ ≥ 0``).
    - With ``variance_type="placebo"``, the summary reports the Fisher p-value and the rank of the observed
      ATT within the placebo distribution (``rank = 1`` = most extreme).
    """

    def __init__(self,
                 unit_col: str = "Unit",
                 time_col: str = "Time",
                 outcome_col: str = "Y",
                 treat_col: str = "W",
                 snn_params: dict | None = None,
                 variance_type: str = "bootstrap",
                 resamples: int = 500,
                 alpha: float = 0.05):
        """
        Initialize the SNN estimator.

        Parameters
        ----------
        unit_col, time_col, outcome_col, treat_col : str, default "Unit", "Time", "Y", "W"
            Column names identifying unit, period, outcome, and treatment.
        snn_params : dict, optional
            Extra keyword arguments forwarded to :class:`SyntheticNearestNeighbors`.
        variance_type : {"jackknife", "bootstrap", "placebo"}, default "bootstrap"
            Method for uncertainty quantification.
        resamples : int, default 500
            Number of resamples (used for ``"bootstrap"``).
        alpha : float, default 0.05
            Significance level for confidence intervals and p-values (jackknife/bootstrap).
        """
        self.unit_col = unit_col
        self.time_col = time_col
        self.outcome_col = outcome_col
        self.treat_col = treat_col
        self.snn_params = snn_params if snn_params is not None else {}
        self.jackknife_se = variance_type == "jackknife"
        self.bootstrap_se = variance_type == "bootstrap"
        self.resamples = resamples
        self.placebo_se = variance_type == "placebo"
        self.alpha = alpha

        # Attributes populated by fit()
        self.results_ = None
        self.overall_att_ = None
        self.att_by_time_ = None
        self.att_by_event_time_ = None
        self.individual_effects_ = None
        self.counterfactual_df_ = None
        self.counterfactual_event_df_ = None
        self.placebo_dist_ = None
        self.placebo_event_dist_ = []
        self._df_proc = None
        self._full_data_treatment_start_map = None
        self._df_effects_all_ = None

    def __repr__(self) -> str:
        """Return a concise string representation of the SNN object."""
        params = [
            f"unit_col='{self.unit_col}'",
            f"time_col='{self.time_col}'",
            f"outcome_col='{self.outcome_col}'",
            f"treat_col='{self.treat_col}'",
            f"variance_type='{'jackknife' if self.jackknife_se else 'bootstrap' if self.bootstrap_se else 'placebo' if self.placebo_se else 'none'}'",
            f"samples={self.resamples}",
            f"alpha={self.alpha}"
        ]
        return f"SNN({', '.join(params)})"

    @staticmethod
    def _get_treatment_start_times(df: pd.DataFrame, unit_col: str, time_col: str, treat_col: str) -> pd.Series:
        """
        Identify the first treatment time for each unit.

        Parameters
        ----------
        df : pandas.DataFrame
            Panel data.
        unit_col : str
            Unit identifier column.
        time_col : str
            Time period column (numeric).
        treat_col : str
            Treatment indicator column (0/1).

        Returns
        -------
        pandas.Series
            Maps each unit ID to its first treatment time. Returns an empty Series
            if no units are treated or if inputs are invalid/non-numeric.
        """
        empty_series_dtype = df[
            time_col].dtype if time_col in df.columns and not df.empty and pd.api.types.is_numeric_dtype(
            df[time_col]) else float
        empty_series = pd.Series(dtype=empty_series_dtype, name='treatment_start_time')

        if df.empty or not all(col in df.columns for col in [unit_col, time_col, treat_col]):
            return empty_series
        if df[treat_col].sum() == 0:
            return empty_series

        temp_df = df.copy()
        if not pd.api.types.is_numeric_dtype(temp_df[time_col]):
            return empty_series

        treat_start_times = temp_df[temp_df[treat_col] == 1].groupby(unit_col)[time_col].min().rename(
            'treatment_start_time')
        return treat_start_times

    def _get_snn_results(self, df_subset: pd.DataFrame, snn_params_from_user: dict | None = None):
        """
        Core internal routine to compute SNN-based treatment effects.

        Workflow:
          1. Pivot the panel to wide outcome/treatment matrices.
          2. Mask treated observations and impute counterfactuals with
             :class:`SyntheticNearestNeighbors`.
          3. Compute individual effects for treated observations and aggregate.

        Parameters
        ----------
        df_subset : pandas.DataFrame
            Subset of the panel data to process.
        snn_params_from_user : dict, optional
            Parameters to pass to the imputer (override defaults).

        Returns
        -------
        tuple
            (individual_effects_df, estimated_overall_att, effects_eventually_treated_df), where

            - individual_effects_df : pandas.DataFrame
                Effects for each treated unit-time observation; columns:
                [unit_col, time_col, "actual", "counterfactual", "effect"].
            - estimated_overall_att : float
                ATT across all treated observations.
            - effects_eventually_treated_df : pandas.DataFrame
                Effects (pre/post) for all units that are ever treated; columns:
                [unit_col, time_col, "effect"].
        """
        empty_individual_effects_df = pd.DataFrame(
            columns=[self.unit_col, self.time_col, 'actual', 'counterfactual', 'effect'])
        empty_effects_etu_df = pd.DataFrame(columns=[self.unit_col, self.time_col, 'effect'])

        if df_subset.empty or df_subset[self.unit_col].nunique() == 0 or df_subset[self.time_col].nunique() == 0:
            return empty_individual_effects_df, np.nan, empty_effects_etu_df

        try:
            if df_subset.duplicated(subset=[self.unit_col, self.time_col]).any():
                df_subset = df_subset.drop_duplicates(subset=[self.unit_col, self.time_col], keep='first')

            outcome_matrix = df_subset.pivot(index=self.unit_col, columns=self.time_col, values=self.outcome_col)
            treatment_matrix = df_subset.pivot(index=self.unit_col, columns=self.time_col,
                                               values=self.treat_col).fillna(0)

            X_masked = outcome_matrix.copy()
            X_masked[treatment_matrix.astype(bool)] = np.nan
            X_masked = X_masked.astype(float)

            if X_masked.empty:
                return empty_individual_effects_df, np.nan, empty_effects_etu_df

            snn_default_internal_params = {'n_neighbors': 1, 'weights': 'distance', 'verbose': False, 'spectral_t': 0.1}
            snn_init_params = snn_default_internal_params.copy()
            if snn_params_from_user:
                snn_init_params.update(snn_params_from_user)

            if X_masked.isnull().all().all() or not X_masked.isnull().any().any():
                imputed_values_matrix = X_masked.copy()
            else:
                snn_model = SyntheticNearestNeighbors(**snn_init_params)
                X_completed_snn_output = snn_model.fit_transform(X_masked)
                if isinstance(X_completed_snn_output, np.ndarray):
                    imputed_values_matrix = pd.DataFrame(X_completed_snn_output, index=outcome_matrix.index,
                                                         columns=outcome_matrix.columns)
                else:
                    imputed_values_matrix = X_completed_snn_output.copy()

            counterfactual_matrix = imputed_values_matrix

            actual_long = outcome_matrix.melt(ignore_index=False, var_name=self.time_col,
                                              value_name='actual').reset_index()
            counterfactual_long = counterfactual_matrix.melt(ignore_index=False, var_name=self.time_col,
                                                             value_name='counterfactual').reset_index()

            df_effects_all = pd.merge(actual_long, counterfactual_long, on=[self.unit_col, self.time_col], how='left')
            df_effects_all['effect'] = df_effects_all['actual'] - df_effects_all['counterfactual']

            treatment_status_long = treatment_matrix.melt(ignore_index=False, var_name=self.time_col,
                                                          value_name=self.treat_col).reset_index()
            df_effects_with_treatment = pd.merge(df_effects_all, treatment_status_long,
                                                 on=[self.unit_col, self.time_col])

            individual_effects_df = df_effects_with_treatment[
                (df_effects_with_treatment[self.treat_col] == 1) & df_effects_with_treatment['actual'].notna()
                ].copy()[[self.unit_col, self.time_col, 'actual', 'counterfactual', 'effect']]

            estimated_overall_att = individual_effects_df['effect'].mean() if not individual_effects_df.empty and \
                                                                              individual_effects_df[
                                                                                  'effect'].notna().any() else np.nan

            eventually_treated_units = df_subset[df_subset[self.treat_col] == 1][self.unit_col].unique()
            effects_eventually_treated_df = pd.DataFrame()
            if len(eventually_treated_units) > 0:
                temp_effects_etu_df = df_effects_all[
                    df_effects_all[self.unit_col].isin(eventually_treated_units)].copy()
                if not temp_effects_etu_df.empty:
                    effects_eventually_treated_df = temp_effects_etu_df[[self.unit_col, self.time_col, 'effect']]
            self._df_effects_all_ = df_effects_all.copy()
            return individual_effects_df, estimated_overall_att, effects_eventually_treated_df
        except Exception:
            return empty_individual_effects_df, np.nan, empty_effects_etu_df

    def _get_event_time_aggregates(self, effects_df: pd.DataFrame, original_panel_df: pd.DataFrame,
                                   treatment_start_map: pd.Series) -> pd.DataFrame:
        """
        Aggregate effects by event time (relative to first treatment).

        Event time is computed using the **order** of unique time periods observed
        for each unit (i.e., sequential index), not absolute differences in time values.

        Parameters
        ----------
        effects_df : pandas.DataFrame
            Effects for eventually treated units; must include [unit_col, time_col, "effect"].
        original_panel_df : pandas.DataFrame
            Full panel dataset used to determine each unit’s observed time sequence.
        treatment_start_map : pandas.Series
            Maps unit IDs to their first treatment time.

        Returns
        -------
        pandas.DataFrame
            Columns: ``["event_time", "att", "N_units"]`` sorted by ``event_time``.
        """
        if effects_df.empty or treatment_start_map.empty:
            return pd.DataFrame(columns=['event_time', 'att', 'N_units'])

        effects_df_valid = effects_df[effects_df['effect'].notna()].copy()
        if effects_df_valid.empty:
            return pd.DataFrame(columns=['event_time', 'att', 'N_units'])

        event_time_data_list = []
        units_in_effects_df = effects_df_valid[self.unit_col].unique()

        for unit_id in units_in_effects_df:
            if unit_id not in treatment_start_map.index:
                continue

            unit_treatment_start_time = treatment_start_map.loc[unit_id]
            unit_observed_times = sorted(
                original_panel_df[original_panel_df[self.unit_col] == unit_id][self.time_col].unique())
            if not unit_observed_times:
                continue

            try:
                treatment_start_index = unit_observed_times.index(unit_treatment_start_time)
            except ValueError:
                # Skip if the start time is not in the observed times
                continue

            unit_specific_effects = effects_df_valid[effects_df_valid[self.unit_col] == unit_id]
            for _, row in unit_specific_effects.iterrows():
                try:
                    current_time_index = unit_observed_times.index(row[self.time_col])
                    sequential_event_time = current_time_index - treatment_start_index
                    event_time_data_list.append({
                        self.unit_col: unit_id, 'event_time': sequential_event_time, 'effect': row['effect']
                    })
                except ValueError:
                    continue

        if not event_time_data_list:
            return pd.DataFrame(columns=['event_time', 'att', 'N_units'])

        sequential_event_time_effects_df = pd.DataFrame(event_time_data_list)
        agg_results = sequential_event_time_effects_df.groupby('event_time').agg(
            att=('effect', 'mean'), N_units=(self.unit_col, 'nunique')
        ).reset_index()
        return agg_results.sort_values('event_time').reset_index(drop=True)

    @staticmethod
    def _calculate_bootstrap_stats(point_estimate, bs_estimates_list, alpha_level):
        """
        Compute SE, p-value, and CI from bootstrap estimates (basic/reverse-percentile).

        The 100(1−α)% CI is:
        ``[2*θ̂ − Q_{1−α/2},  2*θ̂ − Q_{α/2}]``, where ``Q_p`` is the ``p``-th
        percentile of the bootstrap distribution. The p-value uses a normal approximation.

        Parameters
        ----------
        point_estimate : float
            Point estimate from the full sample.
        bs_estimates_list : list or numpy.ndarray
            Bootstrap estimates from resamples.
        alpha_level : float
            Significance level (e.g., 0.05).

        Returns
        -------
        tuple
            ``(se, p_val, ci_low, ci_upp)``.
        """
        valid = np.array([est for est in bs_estimates_list if pd.notna(est) and np.isfinite(est)])
        se = np.nan
        p_val = np.nan
        ci_low = np.nan
        ci_upp = np.nan

        if len(valid) > 1 and pd.notna(point_estimate) and np.isfinite(point_estimate):
            # Standard error
            se = np.std(valid, ddof=0)

            # Normal-approximation p-value
            if se > 1e-9:
                z = point_estimate / se
                p_val = 2 * (1 - scipy.stats.norm.cdf(abs(z)))
            else:
                p_val = 0.0 if abs(point_estimate) > 1e-9 else 1.0

            # Percentiles for basic bootstrap CI
            lower_pct = np.percentile(valid, 100 * (alpha_level / 2))
            upper_pct = np.percentile(valid, 100 * (1 - alpha_level / 2))

            # Basic (reverse-percentile) CI
            ci_low = 2 * point_estimate - upper_pct
            ci_upp = 2 * point_estimate - lower_pct

        return se, p_val, ci_low, ci_upp

    @staticmethod
    def _calculate_jackknife_stats(point_estimate, loo_estimates_list, alpha_level):
        """
        Compute statistics from jackknife (leave-one-out) estimates.

        The CI and p-value use a normal approximation with the jackknife SE.

        Parameters
        ----------
        point_estimate : float
            Original point estimate from the full sample.
        loo_estimates_list : list or numpy.ndarray
            Estimates from each leave-one-out sample.
        alpha_level : float
            Significance level for CI and p-value.

        Returns
        -------
        tuple
            ``(se, p_val, ci_low, ci_upp)``.
        """
        valid_loo = np.array([est for est in loo_estimates_list if pd.notna(est) and np.isfinite(est)])
        N_jack = len(valid_loo)
        se, p_val, ci_low, ci_upp = np.nan, np.nan, np.nan, np.nan
        if N_jack > 1 and pd.notna(point_estimate) and np.isfinite(point_estimate):
            mean_loo = np.mean(valid_loo)
            jack_var = np.sum((valid_loo - mean_loo) ** 2) * (N_jack - 1) / N_jack
            se = np.sqrt(jack_var) if pd.notna(jack_var) and jack_var >= 0 else np.nan
            if pd.notna(se) and se > 1e-9:
                z_score = point_estimate / se
                p_val = 2 * (1 - scipy.stats.norm.cdf(np.abs(z_score)))
                ci_low = point_estimate - scipy.stats.norm.ppf(1 - alpha_level / 2) * se
                ci_upp = point_estimate + scipy.stats.norm.ppf(1 - alpha_level / 2) * se
            elif pd.notna(se):
                p_val = 0.0 if abs(point_estimate) > 1e-9 else 1.0
                ci_low, ci_upp = point_estimate, point_estimate
        return se, p_val, ci_low, ci_upp

    def fit(self, df: pd.DataFrame):
        """
        Fit the SNN model to the provided panel data.

        Parameters
        ----------
        df : pandas.DataFrame
            Panel with required columns ``[unit_col, time_col, outcome_col, treat_col]``.

        Returns
        -------
        SNN
            The fitted instance (for chaining).
        """
        # --- 1. Data Validation and Preparation ---
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input 'df' must be a pandas DataFrame.")

        required_cols = [self.unit_col, self.time_col, self.outcome_col, self.treat_col]
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"DataFrame is missing one or more required columns: {required_cols}")

        df_proc = df[required_cols].copy()
        df_proc = df_proc.drop_duplicates(subset=[self.unit_col, self.time_col], keep='first')

        for col, name in [(self.time_col, "Time"), (self.outcome_col, "Outcome"), (self.treat_col, "Treatment")]:
            if not pd.api.types.is_numeric_dtype(df_proc[col]):
                try:
                    df_proc[col] = pd.to_numeric(df_proc[col], errors='raise' if col != self.outcome_col else 'coerce')
                except ValueError:
                    raise ValueError(f"{name} column '{col}' must be numeric.")

        df_proc = df_proc.dropna(subset=[self.unit_col, self.time_col, self.treat_col])
        self._df_proc = df_proc

        if df_proc.empty or df_proc[self.outcome_col].isnull().all():
            raise ValueError("DataFrame is empty or outcome column has no valid data after cleaning.")

        # --- 2. Core Estimation on Full Data ---
        individual_effects_full, overall_att_full, effects_etu_full_df = \
            self._get_snn_results(df_proc, snn_params_from_user=self.snn_params)

        all_unique_units = df_proc[self.unit_col].unique()
        all_time_periods = sorted(df_proc[self.time_col].unique())
        self._full_data_treatment_start_map = self._get_treatment_start_times(
            df_proc, self.unit_col, self.time_col, self.treat_col
        )
        uniq_starts = self._full_data_treatment_start_map.unique()
        self._common_start_time = uniq_starts[0] if len(uniq_starts) == 1 else None
        treated_units = df_proc[df_proc[self.treat_col] == 1][self.unit_col].unique()

        # Build counterfactual paths
        cf_df = (
            self._df_effects_all_[self._df_effects_all_[self.unit_col].isin(treated_units)]
            .groupby(self.time_col)
            .agg(observed=('actual', 'mean'),
                 counterfactual=('counterfactual', 'mean'))
            .reset_index()
            .rename(columns={self.time_col: 'time'})
        )
        self.counterfactual_df_ = cf_df

        # Event-time counterfactual paths for plotting
        if len(treated_units):
            df_evt = self._df_effects_all_[self._df_effects_all_[self.unit_col].isin(treated_units)].copy()
            df_evt = df_evt.merge(self._full_data_treatment_start_map, left_on=self.unit_col, right_index=True,
                                  how="left")
            df_evt['event_time'] = df_evt[self.time_col] - df_evt['treatment_start_time']
            cf_evt_df = (
                df_evt.groupby('event_time')
                .agg(observed=('actual', 'mean'),
                     counterfactual=('counterfactual', 'mean'),
                     N_units=(self.unit_col, 'nunique'))
                .reset_index()
                .sort_values('event_time')
                .reset_index(drop=True)
            )
        else:
            cf_evt_df = pd.DataFrame(columns=['event_time', 'observed', 'counterfactual', 'N_units'])
        self.counterfactual_event_df_ = cf_evt_df

        # Aggregate by calendar time (ATT)
        att_by_time_full_df = pd.DataFrame({self.time_col: all_time_periods})
        if not effects_etu_full_df.empty and effects_etu_full_df['effect'].notna().any():
            calendar_time_agg = effects_etu_full_df.groupby(self.time_col).agg(
                att=('effect', 'mean'), N_units=(self.unit_col, 'nunique')
            ).reset_index()
            att_by_time_full_df = pd.merge(att_by_time_full_df, calendar_time_agg, on=self.time_col, how='left')
        att_by_time_full_df.fillna({'att': np.nan, 'N_units': 0}, inplace=True)
        att_by_time_full_df['N_units'] = att_by_time_full_df['N_units'].astype(int)

        # Aggregate by event time (ATT)
        att_by_event_time_full_df = self._get_event_time_aggregates(
            effects_etu_full_df, df_proc, self._full_data_treatment_start_map
        )

        # --- 3. Initialize Results Dictionary ---
        self.results_ = {
            'overall_att': {'estimate': overall_att_full, 'method': 'none'},
            'att_by_time': att_by_time_full_df.copy(),
            'att_by_event_time': att_by_event_time_full_df.copy(),
            'individual_effects': individual_effects_full
        }

        # Add inference placeholders; strip later for placebo if needed
        for key in ['overall_att', 'att_by_time', 'att_by_event_time']:
            if isinstance(self.results_[key], dict):
                self.results_[key].update({'se': np.nan, 'p_value': np.nan, 'ci_lower': np.nan, 'ci_upper': np.nan})
            else:
                for col in ['se', 'p_value', 'ci_lower', 'ci_upper']:
                    self.results_[key][col] = np.nan
                self.results_[key]['method'] = 'none'

        # --- 4. Resampling for Standard Errors (if requested) ---
        resampling_snn_params = self.snn_params.copy()
        resampling_snn_params['verbose'] = False

        if self.jackknife_se:
            self._perform_jackknife_inference(all_unique_units, all_time_periods, resampling_snn_params)
        elif self.bootstrap_se:
            self._perform_bootstrap_inference(all_unique_units, all_time_periods, resampling_snn_params)

        if self.placebo_se:
            # Require exactly one treated unit
            ever_treated_by_unit = df_proc.groupby(self.unit_col)[self.treat_col].max()
            if int(ever_treated_by_unit.sum()) != 1:
                raise ValueError("variance_type='placebo' requires exactly one treated unit in the data.")
            self._perform_placebo_inference(all_unique_units, resampling_snn_params)

            # Keep only relevant fields for placebo outputs
            # Overall
            self.results_['overall_att'] = {
                'estimate': self.results_['overall_att']['estimate'],
                'method': 'placebo',
                'placebo_p': self.results_['overall_att'].get('placebo_p', np.nan),
                'placebo_rank': self.results_['overall_att'].get('placebo_rank', np.nan)
            }
            # Event time: keep event_time/att/N_units + placebo_p
            evt = self.results_['att_by_event_time']
            keep_evt_cols = [c for c in ['event_time', 'att', 'N_units', 'placebo_p'] if c in evt.columns]
            self.results_['att_by_event_time'] = evt[keep_evt_cols].copy()
            # Calendar time: drop SE/CI columns if present
            if isinstance(self.results_['att_by_time'], pd.DataFrame):
                drop_cols = [c for c in ['se', 'p_value', 'ci_lower', 'ci_upper', 'method'] if
                             c in self.results_['att_by_time'].columns]
                self.results_['att_by_time'].drop(columns=drop_cols, inplace=True, errors='ignore')

        # --- 5. Finalize and Store Attributes ---
        self.overall_att_ = pd.DataFrame([self.results_['overall_att']])
        self.att_by_time_ = self.results_['att_by_time']
        self.att_by_event_time_ = self.results_['att_by_event_time']
        self.individual_effects_ = self.results_['individual_effects']

        return self

    def _perform_jackknife_inference(self, all_units, all_times, snn_params):
        """
        Run leave-one-unit-out (jackknife) resampling and compute statistics.

        Parameters
        ----------
        all_units : numpy.ndarray
            All unique unit identifiers.
        all_times : list
            Sorted list of unique time periods in the dataset.
        snn_params : dict
            Parameters passed to the internal SNN estimation function.
        """
        if len(all_units) <= 1:
            return

        self.results_['overall_att']['method'] = 'jackknife'
        self.results_['att_by_event_time']['method'] = 'jackknife'

        loo_estimates = {'overall': [], 'by_event_time': []}

        for unit_to_remove in all_units:
            df_loo = self._df_proc[self._df_proc[self.unit_col] != unit_to_remove]
            if df_loo[self.unit_col].nunique() == 0:
                continue

            _, overall_att_j, effects_etu_j = self._get_snn_results(df_loo, snn_params)
            loo_estimates['overall'].append(overall_att_j)

            # Event-time estimates
            treatment_map_loo = self._get_treatment_start_times(df_loo, self.unit_col, self.time_col, self.treat_col)
            att_event_j = self._get_event_time_aggregates(effects_etu_j, df_loo, treatment_map_loo)
            loo_estimates['by_event_time'].append(
                att_event_j.set_index('event_time')['att'] if 'event_time' in att_event_j else pd.Series(dtype=float))

        # Overall ATT stats
        se, p, cil, ciu = self._calculate_jackknife_stats(self.results_['overall_att']['estimate'],
                                                          loo_estimates['overall'], self.alpha)
        self.results_['overall_att'].update({'se': se, 'p_value': p, 'ci_lower': cil, 'ci_upper': ciu})

        # Event-time ATT stats
        all_event_times = sorted(list(set(self.results_['att_by_event_time']['event_time']).union(
            *[s.index for s in loo_estimates['by_event_time']])))
        self._finalize_resampling_stats('jackknife', loo_estimates['by_event_time'], all_event_times,
                                        'att_by_event_time', 'event_time')

    def _perform_bootstrap_inference(self, all_units, all_times, snn_params):
        """
        Run block bootstrap (on units) resampling and compute statistics.

        Parameters
        ----------
        all_units : numpy.ndarray
            All unique unit identifiers.
        all_times : list
            Sorted list of unique time periods in the dataset.
        snn_params : dict
            Parameters passed to the internal SNN estimation function.
        """
        if len(all_units) == 0 or self.resamples < 2:
            return

        self.results_['overall_att']['method'] = 'bootstrap'
        self.results_['att_by_event_time']['method'] = 'bootstrap'

        bs_estimates = {'overall': [], 'by_event_time': []}
        temp_unit_col_bs_base = f"{self.unit_col}_bs_temp"

        for i in range(self.resamples):
            bs_unit_ids = np.random.choice(all_units, size=len(all_units), replace=True)
            temp_unit_col_iter = f"{temp_unit_col_bs_base}_{i}"

            df_bs_rows = []
            for j, unit_id in enumerate(bs_unit_ids):
                unit_data = self._df_proc[self._df_proc[self.unit_col] == unit_id].copy()
                unit_data[temp_unit_col_iter] = f"bs_unit_{j}"
                df_bs_rows.append(unit_data)

            df_bs = pd.concat(df_bs_rows, ignore_index=True) if df_bs_rows else pd.DataFrame()
            if df_bs.empty:
                continue

            # Temporarily switch to the temp unit column for estimation
            original_unit_col = self.unit_col
            self.unit_col = temp_unit_col_iter

            _, overall_att_b, effects_etu_b = self._get_snn_results(df_bs, snn_params)
            bs_estimates['overall'].append(overall_att_b)

            # Event-time estimates
            treatment_map_bs = self._get_treatment_start_times(df_bs, self.unit_col, self.time_col, self.treat_col)
            att_event_b = self._get_event_time_aggregates(effects_etu_b, df_bs, treatment_map_bs)
            bs_estimates['by_event_time'].append(
                att_event_b.set_index('event_time')['att'] if 'event_time' in att_event_b else pd.Series(dtype=float))

            # Revert unit column name
            self.unit_col = original_unit_col

        # Overall ATT stats
        se, p, cil, ciu = self._calculate_bootstrap_stats(self.results_['overall_att']['estimate'],
                                                          bs_estimates['overall'], self.alpha)
        self.results_['overall_att'].update({'se': se, 'p_value': p, 'ci_lower': cil, 'ci_upper': ciu})

        # Event-time ATT stats
        all_event_times = sorted(list(set(self.results_['att_by_event_time']['event_time']).union(
            *[s.index for s in bs_estimates['by_event_time']])))
        self._finalize_resampling_stats('bootstrap', bs_estimates['by_event_time'], all_event_times,
                                        'att_by_event_time', 'event_time')

    def _perform_placebo_inference(self, all_units, snn_params):
        """
        Run deterministic Fisher-style placebo inference (single treated unit only).

        Procedure
        ---------
        1) Require exactly one treated unit in the original data.
        2) Let ``t0`` be that unit’s observed treatment start time.
        3) For each other unit, reassign treatment so that the chosen unit is
           “treated” from ``t0`` onward; everyone else is control.
        4) Re-estimate overall and event-time ATTs for each placebo reassignment.
        5) Append the observed ATT and event-time path to the distributions
           (without recomputation).
        6) Compute:
           - Overall Fisher p-value: share of ATT values (incl. observed)
             with ``|ATT| >= |ATT_obs|``.
           - Per-event-time p-values: for each ``τ``, share of ``ATT_τ`` values
             (incl. observed) with ``|ATT_τ| >= |ATT_obs,τ|``.

        Populates
        ---------
        - ``self.placebo_dist_`` : numpy.ndarray of overall ATT values
        - ``self.placebo_event_dist_`` : list of pandas.Series (event-time paths)
        - ``self.results_['overall_att']`` : adds ``'placebo_p'`` and ``'placebo_rank'``
        - ``self.results_['att_by_event_time']['placebo_p']`` : per-period p-values
        """
        # Count "ever treated" units
        ever_treated_by_unit = self._df_proc.groupby(self.unit_col)[self.treat_col].max()
        n_treated_original = int(ever_treated_by_unit.sum())

        # Enforce exactly one treated unit
        if n_treated_original != 1:
            raise ValueError(
                f"variance_type='placebo' requires exactly one treated unit, but found {n_treated_original}."
            )

        # Identify the single treated unit and its start time
        starts = self._full_data_treatment_start_map
        if starts is None or starts.empty or len(starts) != 1:
            raise ValueError(
                "Could not determine a unique treatment start time for the single treated unit."
            )
        treated_unit = starts.index[0]
        t0 = starts.iloc[0]

        # Prepare list of placebo units (exclude the original treated unit)
        unit_ids = np.array(list(all_units))
        placebo_units = [u for u in unit_ids if u != treated_unit]

        placebo_atts = []
        placebo_evt_series = []

        # Run placebo reassignments for each placebo unit
        for u in placebo_units:
            df_perm = self._df_proc.copy()
            # All control by default
            df_perm[self.treat_col] = 0
            # Placebo treatment for unit u from t0 forward
            df_perm.loc[
                (df_perm[self.unit_col] == u) & (df_perm[self.time_col] >= t0),
                self.treat_col
            ] = 1

            # Re-estimate outcomes and compute placebo ATT + event-time path
            _, att_perm, effects_etu_perm = self._get_snn_results(df_perm, snn_params)
            placebo_atts.append(att_perm)

            placebo_start_map = self._get_treatment_start_times(
                df_perm, self.unit_col, self.time_col, self.treat_col
            )
            evt = self._get_event_time_aggregates(effects_etu_perm, df_perm, placebo_start_map)
            placebo_evt_series.append(evt.set_index('event_time')['att'])

        # Append the observed configuration to the distributions (no recomputation)
        obs_att = self.results_['overall_att']['estimate']
        obs_evt_df = self.results_.get('att_by_event_time', pd.DataFrame())
        obs_evt_series = (obs_evt_df.set_index('event_time')['att']
                          if not obs_evt_df.empty and 'att' in obs_evt_df.columns else pd.Series(dtype=float))

        # Overall distribution
        dist = np.array([p for p in placebo_atts if pd.notna(p) and np.isfinite(p)], dtype=float)
        if pd.notna(obs_att) and np.isfinite(obs_att):
            dist = np.append(dist, obs_att)
        self.placebo_dist_ = dist

        # Event-time distribution
        self.placebo_event_dist_ = placebo_evt_series + ([obs_evt_series] if not obs_evt_series.empty else [])

        # Overall Fisher-style p-value and rank
        if np.isnan(obs_att) or len(self.placebo_dist_) == 0:
            p_val_overall, rank = np.nan, np.nan
        else:
            p_val_overall = float(np.mean(np.abs(self.placebo_dist_) >= abs(obs_att)))
            rank = 1 + np.sum(np.abs(self.placebo_dist_) > abs(obs_att))

        self.results_['overall_att']['placebo_p'] = p_val_overall
        self.results_['overall_att']['placebo_rank'] = rank

        # Per-event-time p-values
        if not obs_evt_df.empty and self.placebo_event_dist_:
            # Collect event-time values into a DataFrame
            placebo_evt_df = pd.concat(
                [s.rename(i) for i, s in enumerate(self.placebo_event_dist_)],
                axis=1
            )
            placebo_evt_df.index.name = 'event_time'

            per_period_p = []
            for _, row in obs_evt_df.iterrows():
                tau = row['event_time']
                if tau in placebo_evt_df.index:
                    dist_tau = placebo_evt_df.loc[tau].dropna().values
                    if dist_tau.size:
                        p_tau = float(np.mean(np.abs(dist_tau) >= abs(row['att'])))
                    else:
                        p_tau = np.nan
                else:
                    p_tau = np.nan
                per_period_p.append(p_tau)

            # Keep only relevant columns and attach p-values
            keep_cols = [c for c in ['event_time', 'att', 'N_units'] if c in obs_evt_df.columns]
            obs_evt_df = obs_evt_df[keep_cols].copy()
            obs_evt_df['placebo_p'] = per_period_p
            self.results_['att_by_event_time'] = obs_evt_df

    def _finalize_resampling_stats(self, method, estimates_list, all_indices, results_key, index_col):
        """
        Compute and merge SE/CI/p-values for by-time or by-event-time results.

        Parameters
        ----------
        method : {"jackknife", "bootstrap"}
            Resampling method used.
        estimates_list : list[pandas.Series]
            Each element is a Series of estimates from a single resample,
            indexed by time or event time.
        all_indices : list
            Union of indices (time/event_time) across all resamples.
        results_key : str
            Key in ``self.results_`` to update (e.g., ``"att_by_event_time"``).
        index_col : str
            Name of the index column (e.g., ``"Time"`` or ``"event_time"``).
        """
        if not estimates_list:
            return

        reindexed_series = [s.reindex(all_indices, fill_value=np.nan) for s in estimates_list]
        resampling_df = pd.concat(reindexed_series, axis=1).T

        stats_list = []
        target_df = self.results_[results_key]

        for _, row in target_df.iterrows():
            point_est = row['att']
            resampled_ests = resampling_df[row[index_col]].tolist() if row[index_col] in resampling_df.columns else []

            if method == 'jackknife':
                stats_list.append(self._calculate_jackknife_stats(point_est, resampled_ests, self.alpha))
            else:  # bootstrap
                stats_list.append(self._calculate_bootstrap_stats(point_est, resampled_ests, self.alpha))

        if stats_list:
            stats_df = pd.DataFrame(stats_list, columns=['se', 'p_value', 'ci_lower', 'ci_upper'],
                                    index=target_df.index)
            for col in stats_df.columns:
                target_df[col] = stats_df[col]

    def plot(self,
             plot_type: str = "gap",
             calendar_time: bool = False,
             show_placebos: bool = False,
             xrange: tuple | None = None,
             title: str | None = None,
             xlabel: str | None = None,
             ylabel: str = "Outcome",
             figsize: tuple = (10, 6),
             color: str = "#33658A",
             observed_color: str = "#070707",
             counterfactual_color: str = "#33658A",
             placebo_color: str = "#999999",
             placebo_opacity: float = 0.25,
             vertical_line_color: str = "#E71D36") -> go.Figure:
        """
        Generate diagnostic plots of estimation results.

        Two plot types (both default to event time on the x-axis):

        - ``plot_type="gap"``: ATT series (the “gap” plot).
        - ``plot_type="counterfactual"``: Observed vs. counterfactual outcome
          paths for treated units.

        If treatment adoption is simultaneous across units, setting
        ``calendar_time=True`` will display the x-axis in calendar time rather
        than event time. For staggered adoption, this raises an error.

        Parameters
        ----------
        plot_type : {"gap", "counterfactual"}, default "gap"
            Type of plot.
        calendar_time : bool, default False
            Use calendar time on the x-axis (only allowed for simultaneous adoption).
        show_placebos : bool, default False
            Overlay placebo distribution lines in the gap plot (ignored for counterfactual plot).
        xrange : tuple[int, int] or None, optional
            ``(min, max)`` x-axis range filter.
        title : str or None, optional
            Plot title; a sensible default is used if ``None``.
        xlabel : str or None, optional
            X-axis label; a sensible default is used if ``None``.
        ylabel : str, default "Outcome"
            Y-axis label (for gap, defaults to “ATT” internally).
        figsize : tuple, default (10, 6)
            Figure size in inches; internally scaled by 100 for pixels.
        color : str, default "#33658A"
            Line/marker color for the ATT in the gap plot.
        observed_color : str, default "#070707"
            Observed series color in the counterfactual plot.
        counterfactual_color : str, default "#33658A"
            Counterfactual series color in the counterfactual plot.
        placebo_color : str, default "#999999"
            Placebo lines color (gap plot).
        placebo_opacity : float, default 0.25
            Placebo lines opacity (gap plot).
        vertical_line_color : str, default "#E71D36"
            Color for the vertical line marking treatment adoption.

        Returns
        -------
        plotly.graph_objects.Figure
            The generated figure.
        """
        if show_placebos and plot_type == "counterfactual":
            raise ValueError("show_placebos=True is not supported for counterfactual plots.")

        # helper for CI fill
        def _hex_to_rgba(hx: str, alpha: float):
            hx = hx.lstrip('#')
            r, g, b = (int(hx[i: i + 2], 16) for i in (0, 2, 4))
            return f"rgba({r},{g},{b},{alpha})"

        # --- Define plot settings based on calendar_time flag ---
        x_axis_col = 'event_time'
        default_xlabel = "Event time τ"
        default_title_suffix = "(event time)"
        hover_prefix = "τ"
        vline_x = -0.5

        if calendar_time:
            if getattr(self, "_common_start_time", None) is None:
                raise ValueError(
                    "calendar_time=True is only allowed for simultaneous treatment adoption.\n"
                    "This dataset appears to have staggered adoption."
                )
            x_axis_col = 'calendar_x'
            default_xlabel = "Calendar time t"
            default_title_suffix = "(calendar time)"
            hover_prefix = "t"
            vline_x = float(self._common_start_time) - 0.5

        # --- GAP plot ---
        if plot_type == "gap":
            if self.att_by_event_time_ is None:
                raise RuntimeError("Call .fit() before plotting.")
            df = self.att_by_event_time_.copy()
            if calendar_time:
                df[x_axis_col] = df['event_time'] + self._common_start_time
            if xrange is not None and len(xrange) == 2:
                df = df[df[x_axis_col].between(*xrange)]

            fig = make_subplots()
            # ATT line + markers
            fig.add_trace(go.Scatter(
                x=df[x_axis_col], y=df['att'],
                mode="lines+markers", name="ATT",
                line=dict(color=color, width=2),
                marker=dict(size=7, color=color),
                hovertemplate=f"{hover_prefix}=%{{x}}: %{{y:.3f}}<extra></extra>"
            ))

            if show_placebos and getattr(self, "placebo_event_dist_", []):
                for s in self.placebo_event_dist_:
                    if s.empty:
                        continue
                    fig.add_trace(go.Scatter(
                        x=s.index, y=s.values,
                        mode="lines",
                        line=dict(color=placebo_color, width=1),
                        opacity=placebo_opacity,
                        hoverinfo="skip",
                        showlegend=False
                    ))

            # CI band
            if {'ci_lower', 'ci_upper'}.issubset(df.columns):
                ci = df.dropna(subset=['ci_lower', 'ci_upper'])
                if not ci.empty:
                    fig.add_trace(go.Scatter(
                        x=list(ci[x_axis_col]) + ci[x_axis_col].tolist()[::-1],
                        y=list(ci['ci_upper']) + ci['ci_lower'].tolist()[::-1],
                        fill="toself",
                        fillcolor=_hex_to_rgba(color, 0.25),
                        line=dict(color="rgba(0,0,0,0)"), name="95% CI",
                        hoverinfo="skip"
                    ))

            fig.add_hline(y=0, line_dash="dash", line_width=1)
            if (xrange is not None and xrange[0] < 0 + self._common_start_time*calendar_time < xrange[1]) or xrange is None:
                fig.add_vline(x=vline_x, line_dash="dot", line_color=vertical_line_color, line_width=1)

            fig.update_layout(
                template="plotly_white",
                width=figsize[0] * 100, height=figsize[1] * 100,
                title=title or f"ATT {default_title_suffix}",
                xaxis_title=xlabel or default_xlabel,
                yaxis_title=ylabel or "ATT",
                legend=dict(orientation="h", x=0.5, y=-0.18, xanchor="center"),
                margin=dict(l=60, r=40, t=80, b=80)
            )

            # Auto-pad y-axis to full CI range
            if {'ci_lower', 'ci_upper'}.issubset(df.columns):
                ci_only = df.dropna(subset=['ci_lower', 'ci_upper'])
                if not ci_only.empty:
                    ymin, ymax = ci_only['ci_lower'].min(), ci_only['ci_upper'].max()
                    pad = 0.05 * (ymax - ymin)
                    fig.update_yaxes(range=[ymin - pad, ymax + pad])

            fig.show()
            return fig

        # --- COUNTERFACTUAL plot ---
        if plot_type == "counterfactual":
            df = getattr(self, "counterfactual_event_df_", None)
            if df is None or df.empty:
                raise RuntimeError("counterfactual_event_df_ not found – run .fit() first.")
            df = df.copy()

            if calendar_time:
                df[x_axis_col] = df['event_time'] + self._common_start_time

            # Bring in ATT CI to translate into counterfactual CI
            ci_data = self.att_by_event_time_[['event_time', 'att', 'ci_lower', 'ci_upper']]
            df = df.merge(ci_data, on='event_time', how='left')
            df['cf_lower'] = df['counterfactual'] - (df['ci_upper'] - df['att'])
            df['cf_upper'] = df['counterfactual'] + (df['att'] - df['ci_lower'])

            if xrange is not None and len(xrange) == 2:
                df = df[df[x_axis_col].between(*xrange)]

            fig = go.Figure()

            # Counterfactual CI band
            band = df.dropna(subset=['cf_lower', 'cf_upper'])
            if not band.empty:
                fig.add_trace(go.Scatter(
                    x=list(band[x_axis_col]) + band[x_axis_col].tolist()[::-1],
                    y=list(band['cf_upper']) + band['cf_lower'].tolist()[::-1],
                    fill="toself",
                    fillcolor=_hex_to_rgba(counterfactual_color, 0.25),
                    line=dict(color="rgba(0,0,0,0)"), name="95% CI",
                    hoverinfo="skip"
                ))

            # Observed & counterfactual lines
            fig.add_trace(go.Scatter(
                x=df[x_axis_col], y=df['observed'],
                mode="lines", name="Observed",
                line=dict(color=observed_color, width=2)
            ))
            fig.add_trace(go.Scatter(
                x=df[x_axis_col], y=df['counterfactual'],
                mode="lines", name="Counterfactual",
                line=dict(color=counterfactual_color, width=2, dash="dash")
            ))
            if (xrange is not None and xrange[0] < 0 + self._common_start_time*calendar_time < xrange[1]) or xrange is None:
                fig.add_vline(x=vline_x, line_dash="dot", line_color=vertical_line_color, line_width=1)

            fig.update_layout(
                template="plotly_white",
                width=figsize[0] * 100, height=figsize[1] * 100,
                title=title or f"Observed vs Synthetic {default_title_suffix}",
                xaxis_title=xlabel or default_xlabel,
                yaxis_title=ylabel,
                legend=dict(orientation="h", x=0.5, y=-0.18, xanchor="center"),
                margin=dict(l=60, r=40, t=80, b=80)
            )

            # Pad y-axis to include both CF and observed ranges
            non_na = df.dropna(subset=['cf_lower', 'cf_upper', 'observed', 'counterfactual'])
            if not non_na.empty:
                ymin = min(non_na['cf_lower'].min(), non_na['observed'].min())
                ymax = max(non_na['cf_upper'].max(), non_na['observed'].max())
                pad = 0.05 * (ymax - ymin)
                fig.update_yaxes(range=[ymin - pad, ymax + pad])

            fig.show()
            return fig

        raise ValueError("`plot_type` must be one of 'gap' or 'counterfactual'.")

    def summary(self) -> None:
        """
        Print a formatted summary of the estimation results.
        """
        if self.results_ is None:
            raise RuntimeError("You must call the .fit() method before generating a summary.")

        print("=" * 60)
        print("SNN Estimation Results")
        print("=" * 60)

        # -------- Overall ATT --------
        if self.placebo_se:
            # Only the relevant placebo fields
            overall_summary_df = pd.DataFrame([{
                'estimate': self.overall_att_.iloc[0]['estimate'],
                'placebo_p': self.overall_att_.iloc[0].get('placebo_p', np.nan),
                'placebo_rank': self.overall_att_.iloc[0].get('placebo_rank', np.nan),
            }])
            # Pretty format
            for col in ['estimate', 'placebo_p']:
                overall_summary_df[col] = overall_summary_df[col].map(lambda x: f'{x:.4g}' if pd.notna(x) else 'nan')
            if 'placebo_rank' in overall_summary_df.columns:
                overall_summary_df['placebo_rank'] = overall_summary_df['placebo_rank'].apply(
                    lambda x: str(int(x)) if pd.notna(x) else 'nan'
                )
        else:
            overall_summary_df = self.overall_att_.copy()
            float_cols = overall_summary_df.select_dtypes(include=np.number).columns
            overall_summary_df[float_cols] = overall_summary_df[float_cols].map(
                lambda x: f'{x:.4g}' if pd.notna(x) else 'nan')

        print("\n--- Overall ATT ---")
        print(overall_summary_df.to_string(index=False))

        # Placebo overall Fisher p-value line (for clarity)
        if self.placebo_se:
            res = self.results_['overall_att']
            p = res.get('placebo_p', np.nan)
            rank = res.get('placebo_rank', np.nan)
            m = len(self.placebo_dist_) if self.placebo_dist_ is not None else np.nan
            print(f"\nPlacebo Fisher p-value: {p:.4g}  "
                  f"(rank {int(rank) if pd.notna(rank) else 'N/A'}/{int(m) if pd.notna(m) else 'N/A'})")

        # -------- Event-time table --------
        print("\n\n--- ATT by Event Time (Post-Treatment) ---\n")
        evt_df = self.att_by_event_time_.copy()

        # Only non-negative event times
        if 'event_time' in evt_df.columns:
            evt_df = evt_df[evt_df['event_time'] >= 0].copy()

        if self.placebo_se:
            # Keep relevant columns and pretty format
            keep_cols = [c for c in ['event_time', 'att', 'N_units', 'placebo_p'] if c in evt_df.columns]
            evt_df = evt_df[keep_cols]
            num_cols = [c for c in ['att', 'placebo_p'] if c in evt_df.columns]
            for c in num_cols:
                evt_df[c] = evt_df[c].map(lambda x: f'{x:.4g}' if pd.notna(x) else 'nan')
            if 'N_units' in evt_df.columns:
                evt_df['N_units'] = evt_df['N_units'].astype(str)
        else:
            float_cols_evt = evt_df.select_dtypes(include=np.number).columns
            evt_df[float_cols_evt] = evt_df[float_cols_evt].map(lambda x: f'{x:.4g}' if pd.notna(x) else 'nan')
            if 'N_units' in evt_df.columns:
                evt_df['N_units'] = self.att_by_event_time_['N_units'].astype(str)

        print(evt_df.to_string(index=False))
        print("\n" + "=" * 60)
