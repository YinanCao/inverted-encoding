# Inverted encoding model (IEM) for MEG localizer data

MATLAB code for an inverted encoding model on MEG sensor data from a sensory localizer, with leave-one-run-out cross-validation. A stimulus appears at one of 3 locations (0, 120, 240 deg). The model estimates sensor weights for 3 location channels from the training runs, inverts them on the held-out run, and decodes the shown location from the resulting channel responses.

## Files

- `IEM_localizer_crossval.m`: main script, runs one subject end to end and plots the results.
- `covdiag_shrink.m`: noise covariance estimator shrunk towards its diagonal (Kriegeskorte and Stokes).
- `localizerData/`: input data, one slim file per subject (`*_SUBJ_<n>_SL_slim.mat`).
- `IEMresults/`: output, `IEM_crossval_SUBJ_<n>.mat`.

## Requirements

Base MATLAB only. The script uses `yline`, `xline` and `sgtitle`, so R2018b or newer.

## Usage

1. Put the subject's slim localizer file in `localizerData/`.
2. Set `subjID` at the top of `IEM_localizer_crossval.m`.
3. Run the script. Keep `covdiag_shrink.m` in the same folder.

## Input

Each slim file holds `meg_trl` (sensor x time x trial, -100 to 500 ms, 100 Hz), `time` (s, 0 = stimulus onset), `trlinfo` (stimulus trigger, run index) and `designM` (design table, last column is the trigger).

## Method

1. Remove flicker (catch) trials, triggers 29 to 31, and trials rejected as NaN.
2. Baseline correct each trial and sensor using -100 to 0 ms. Analyse up to 450 ms.
3. Build the channel design matrix with basis functions `|cos(0.5*(theta - mu))|^5`.
4. For each time point and each held-out run, fit encoding weights by least squares, estimate the shrunk noise covariance of the training residuals, and compute noise-whitened inverse weights. Centring uses the training mean only, so the test run does not leak into the fit.
5. Apply the inverse weights to the held-out run to get channel responses.
6. Read out the location as a population vector and score a trial as correct if the shown location has the largest value. Chance is 1/3.

## Output

`IEM_crossval_SUBJ_<n>.mat` contains:

- `decoding_acc`: time x read-out (cosine similarity, projection, raw channel) x stimulus type (top, left, right, all).
- `IEM_crossval`: channel responses, channel x time x trial.
- `timelist`, `trldesign`, `IEMconfig`, `dim_info`.

The script also plots decoding accuracy over time and the mean channel responses at 140 to 170 ms. With 3 equally spaced channels the three read-outs (cosine similarity, projection, raw channel response) give identical accuracy.

## Notes

The method follows `SL_crossVal_perm.m` (MEGframing2022) without the permutation test. Two intended differences: demeaning is done inside each fold, and NaN trials are also removed from the design labels.
