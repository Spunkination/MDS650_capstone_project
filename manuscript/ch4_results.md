# Chapter 4: Analysis, Experiments and Results

All results come from the clean Apple Silicon execution of Section 3.5, using the common 2025 test window throughout. Neural headline metrics score the average of five seed forecasts; individual-seed dispersion is reported separately.

## 4.1 Data Characteristics

The cleaned series contains 46,570 complete hours. Generation is zero-inflated, strongly diurnal and seasonally structured. Capacity growth and market constraints make the target non-stationary, so these plots describe market-cleared generation rather than a fixed physical production distribution.

% [FIGURE: eda/actual_generation_timeline.png]

% [FIGURE: eda/target_distribution.png]

Mean output follows the solar day, while seasonal profile width and level vary across the year. The split diagnostic shows 2025 is not exchangeable with earlier years, supporting chronological rather than random evaluation.

% [FIGURE: eda/hourly_seasonal_patterns.png]

% [FIGURE: eda/target_split_drift.png]

Measured output also trends upward, which matters for comparability between splits.

% [FIGURE: observed_output_scale.png]

The five-minute source retains fluctuations hidden by hourly averaging. Its scale-normalised coefficient of variation and relative range are the candidate incremental predictors; their relationship with the next-hour ramp motivates the experiment but is not causal evidence.

% [FIGURE: eda/intra_hour_variability.png]

% [FIGURE: eda/intra_hour_signal_next_ramp.png]

The exploratory sequence supports three decisions. Zero output is retained because night-time behaviour belongs to the forecasting problem and the physics-guided loss. Visible split drift justifies past-only scaling instead of a full-sample maximum, and the monotone rise in next-hour change across variability bins makes an incremental short-horizon signal plausible. None of this selects a winning model; that rests on the held-out paired comparisons.

## 4.2 Baseline Performance

Persistence is competitive at 24 hours through daily periodicity but weak at 3 and 6 hours across the daytime rise and fall. SARIMA is the strongest baseline at 1-6 hours; rolling climatology is stable across horizons.

The magnitude of the contrast is substantial. Persistence RMSE rises from 469.06 MW at one hour to 1,631.92 MW at six hours, then falls to 423.60 MW at 24 hours. This U-shape reflects the difference between copying the latest output across the intraday solar cycle and copying it to the same time the next day. Rolling climatology stays close to 571 MW because its same-hour average does not change its information with horizon. SARIMA reaches 221.69 MW at one hour and 433.32 MW at three hours, making a target-only time-series model a demanding comparator without weather fields.

Table 4.1: Baseline performance on the common 2025 test window. RMSE and MAE are MW.

| Model | h | RMSE | MAE | R2 | Persistence skill (%) |
| --- | --- | --- | --- | --- | --- |
| Persistence | 1 | 469.06 | 242.10 | 0.84 | 0.00 |
| Persistence | 3 | 1092.77 | 658.82 | 0.13 | 0.00 |
| Persistence | 6 | 1631.92 | 1210.27 | -0.94 | 0.00 |
| Persistence | 24 | 423.60 | 219.52 | 0.87 | 0.00 |
| Rolling climatology | 1 | 570.93 | 348.28 | 0.76 | -21.72 |
| Rolling climatology | 3 | 570.99 | 348.36 | 0.76 | 47.75 |
| Rolling climatology | 6 | 571.09 | 348.47 | 0.76 | 65.01 |
| Rolling climatology | 24 | 570.37 | 347.76 | 0.76 | -34.65 |
| SARIMA | 1 | 221.69 | 118.13 | 0.96 | 52.74 |
| SARIMA | 3 | 433.32 | 245.14 | 0.86 | 60.35 |
| SARIMA | 6 | 542.53 | 343.98 | 0.79 | 66.76 |
| SARIMA | 24 | 634.49 | 330.15 | 0.71 | -49.78 |

Baseline skill relative to persistence must therefore be read with the horizon. Rolling climatology is 21.72 percent worse at one hour but 65.01 percent better at six hours. SARIMA improves by 52.74-66.76 percent at one to six hours yet is 49.78 percent worse at 24 hours. A single baseline or horizon average would conceal these reversals.

% [FIGURE: baseline_comparison_rmse.png | space-before: 12pt]

## 4.3 Learned-Model Performance

On combined inputs, LSTM gives the lowest 1-hour RMSE, while the physics-guided CNN-BiLSTM is best at 3 and 6 hours. The unconstrained CNN-BiLSTM is best at 24 hours. XGBoost beats persistence substantially at 1-6 hours but not at 24 hours.

Absolute error grows with horizon for every learned model, but relative skill does not change monotonically. XGBoost RMSE increases from 230.70 to 410.07 MW between one and six hours, while its persistence skill rises from 50.82 to 74.87 percent because persistence deteriorates faster. At 24 hours learned RMSE spans 399.77 to 435.82 MW and persistence skill ranges from -2.88 to 5.63 percent.

Table 4.2: Combined-input learned-model and ensemble metrics. RMSE, MAE, bias and daylight RMSE are MW.

| Model | h | RMSE | Relative RMSE (%) | MAE | Bias | R2 | Daylight RMSE | Skill (%) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| XGBoost | 1 | 230.70 | 6.88 | 119.82 | -69.47 | 0.96 | 329.57 | 50.82 |
| LSTM | 1 | 155.25 | 4.70 | 83.68 | -9.05 | 0.98 | 220.43 | 66.90 |
| CNN-BiLSTM | 1 | 155.58 | 4.71 | 85.98 | -16.53 | 0.98 | 221.50 | 66.83 |
| Physics-guided | 1 | 155.53 | 4.71 | 86.09 | -18.53 | 0.98 | 221.46 | 66.84 |
| Ensemble | 1 | 161.39 | 4.87 | 88.57 | -26.82 | 0.98 | 230.07 | 65.59 |
| XGBoost | 3 | 344.49 | 10.42 | 190.01 | -110.58 | 0.91 | 492.09 | 68.48 |
| LSTM | 3 | 302.65 | 9.16 | 165.90 | -52.09 | 0.93 | 430.89 | 72.30 |
| CNN-BiLSTM | 3 | 300.92 | 9.14 | 169.11 | -51.13 | 0.93 | 427.31 | 72.46 |
| Physics-guided | 3 | 300.28 | 9.12 | 170.12 | -50.85 | 0.93 | 426.52 | 72.52 |
| Ensemble | 3 | 303.88 | 9.21 | 169.34 | -65.49 | 0.93 | 432.94 | 72.19 |
| XGBoost | 6 | 410.07 | 12.43 | 228.15 | -148.87 | 0.88 | 585.83 | 74.87 |
| LSTM | 6 | 375.28 | 11.37 | 216.11 | -90.88 | 0.90 | 533.77 | 77.00 |
| CNN-BiLSTM | 6 | 373.57 | 11.34 | 218.92 | -85.29 | 0.90 | 529.89 | 77.11 |
| Physics-guided | 6 | 369.17 | 11.20 | 217.28 | -78.09 | 0.90 | 523.53 | 77.38 |
| Ensemble | 6 | 373.77 | 11.33 | 215.43 | -100.16 | 0.90 | 532.19 | 77.10 |
| XGBoost | 24 | 435.82 | 13.20 | 248.83 | -132.44 | 0.86 | 622.48 | -2.88 |
| LSTM | 24 | 414.86 | 12.59 | 239.90 | -121.68 | 0.87 | 591.80 | 2.06 |
| CNN-BiLSTM | 24 | 399.77 | 12.16 | 232.06 | -93.88 | 0.88 | 569.90 | 5.63 |
| Physics-guided | 24 | 400.81 | 12.19 | 231.78 | -95.44 | 0.88 | 571.66 | 5.38 |
| Ensemble | 24 | 407.40 | 12.37 | 235.35 | -110.52 | 0.88 | 581.59 | 3.83 |

% [FIGURE: forecast_skill_by_model.png]

% [FIGURE: learned_model_error_diagnostics.png]

The additional metrics agree with the RMSE ordering. Bias is negative for every configuration and largest for XGBoost at each horizon. Daylight RMSE exceeds all-hour RMSE because retained night-time zeros are easy, so no scalar score represents uniform performance across the operating day.

Zero clipping also forms part of the measurement chain. At one hour, 16.71 percent of LSTM and physics-guided, 19.47 percent of CNN-BiLSTM and 8.38 percent of XGBoost outputs are clipped. Much clipping falls at night, where zero is appropriate, but it can mask an unconstrained negative prediction.

The ranking must be qualified by paired inference. All neural procedures robustly outperform XGBoost at 1, 3 and 6 hours. Both CNN-BiLSTM variants also do so at 24 hours, while LSTM does not after correction. Among neural procedures, the two CNN-BiLSTM variants outperform LSTM robustly only at 24 hours.

The matched comparison isolates the physical penalties. Their point gain is 0.04 percent at 1 hour, 0.21 at 3 hours, 1.18 at 6 hours and -0.26 at 24 hours. Only the 6-hour gain is robust; the other horizons do not establish a penalty benefit or harm.

% [FIGURE: pairwise_model_superiority.png]

Table 4.3: Pairwise learned-model RMSE gain of the second model over the first and robust conclusion.

| Comparison | h | Gain (%) | Robust conclusion |
| --- | --- | --- | --- |
| XGBoost vs LSTM | 1 | 32.71 | LSTM better |
| XGBoost vs LSTM | 3 | 12.15 | LSTM better |
| XGBoost vs LSTM | 6 | 8.49 | LSTM better |
| XGBoost vs LSTM | 24 | 4.81 | No robust difference |
| XGBoost vs CNN-BiLSTM | 1 | 32.56 | CNN-BiLSTM better |
| XGBoost vs CNN-BiLSTM | 3 | 12.65 | CNN-BiLSTM better |
| XGBoost vs CNN-BiLSTM | 6 | 8.90 | CNN-BiLSTM better |
| XGBoost vs CNN-BiLSTM | 24 | 8.27 | CNN-BiLSTM better |
| XGBoost vs Physics-guided | 1 | 32.58 | Physics-guided better |
| XGBoost vs Physics-guided | 3 | 12.83 | Physics-guided better |
| XGBoost vs Physics-guided | 6 | 9.98 | Physics-guided better |
| XGBoost vs Physics-guided | 24 | 8.03 | Physics-guided better |
| LSTM vs CNN-BiLSTM | 1 | -0.22 | No robust difference |
| LSTM vs CNN-BiLSTM | 3 | 0.57 | No robust difference |
| LSTM vs CNN-BiLSTM | 6 | 0.46 | No robust difference |
| LSTM vs CNN-BiLSTM | 24 | 3.64 | CNN-BiLSTM better |
| LSTM vs Physics-guided | 1 | -0.18 | No robust difference |
| LSTM vs Physics-guided | 3 | 0.78 | No robust difference |
| LSTM vs Physics-guided | 6 | 1.63 | No robust difference |
| LSTM vs Physics-guided | 24 | 3.39 | Physics-guided better |
| CNN-BiLSTM vs Physics-guided | 1 | 0.04 | No robust difference |
| CNN-BiLSTM vs Physics-guided | 3 | 0.21 | No robust difference |
| CNN-BiLSTM vs Physics-guided | 6 | 1.18 | Physics-guided better |
| CNN-BiLSTM vs Physics-guided | 24 | -0.26 | No robust difference |

A separate finding concerns optimisation rather than information: the epoch minimising validation error falls sharply as the horizon lengthens, and the day-ahead fits stop improving well before the intraday fits.

Table 4.4: Mean best-validation epoch across the five seeds, by model, feature set and horizon. Lower values mean the fit stopped improving sooner.

| Model | Inputs | 1h | 3h | 6h | 24h |
| --- | --- | --- | --- | --- | --- |
| LSTM | combined | 61.2 | 31.0 | 18.4 | 8.8 |
| LSTM | weather_history | 51.2 | 39.2 | 20.2 | 11.6 |
| CNN-BiLSTM | combined | 54.8 | 23.8 | 22.6 | 12.0 |
| CNN-BiLSTM | weather_history | 63.4 | 22.0 | 22.8 | 12.0 |
| Physics-guided CNN-BiLSTM | combined | 62.0 | 23.4 | 21.6 | 12.4 |
| Physics-guided CNN-BiLSTM | weather_history | 60.0 | 26.2 | 22.4 | 12.0 |

The one-hour history-based fits improve for roughly 51 to 63 epochs, whereas the day-ahead fits stop improving within about 9 to 12 epochs. This concerns optimisation under the fixed schedule, not the information content of the inputs.

## 4.4 Controlled Contribution of Intra-Hour Features

Without ordinary history, the two scale-normalised summaries improve every learned procedure at every horizon under the full decision rule. Gains range from 1.26 to 11.46 percent. Agreement across four different procedures, weekly-block intervals, corrected tests and the output-scale sensitivity strengthens the conclusion that the summaries carry predictive information beyond retrospective meteorology alone.

Table 4.5: Intra-hour contribution without hourly history. Positive gain favours `weather_intra`; confidence intervals use 168-hour blocks.

| Model | h | RMSE weather | RMSE weather_intra | Gain (%) | 95% CI (%) | Scale gain (%) |
| --- | --- | --- | --- | --- | --- | --- |
| XGBoost | 1 | 722.73 | 690.18 | 4.50 | 3.15 to 6.13 | 4.68 |
| XGBoost | 3 | 729.24 | 707.76 | 2.95 | 1.94 to 4.10 | 3.08 |
| XGBoost | 6 | 730.86 | 714.90 | 2.18 | 1.29 to 3.19 | 2.31 |
| XGBoost | 24 | 746.36 | 736.95 | 1.26 | 0.67 to 1.91 | 1.31 |
| LSTM | 1 | 695.51 | 615.80 | 11.46 | 8.62 to 14.11 | 11.32 |
| LSTM | 3 | 690.91 | 635.16 | 8.07 | 6.12 to 9.95 | 8.16 |
| LSTM | 6 | 695.79 | 653.63 | 6.06 | 3.80 to 8.37 | 6.07 |
| LSTM | 24 | 722.51 | 664.96 | 7.97 | 5.52 to 10.33 | 7.86 |
| CNN-BiLSTM | 1 | 674.77 | 615.90 | 8.72 | 6.61 to 11.09 | 8.81 |
| CNN-BiLSTM | 3 | 687.31 | 642.52 | 6.52 | 5.04 to 8.18 | 6.56 |
| CNN-BiLSTM | 6 | 690.28 | 651.55 | 5.61 | 4.18 to 7.00 | 5.62 |
| CNN-BiLSTM | 24 | 708.47 | 651.34 | 8.06 | 6.63 to 9.66 | 8.24 |
| Physics-guided | 1 | 677.33 | 608.46 | 10.17 | 7.89 to 12.62 | 10.19 |
| Physics-guided | 3 | 687.92 | 635.70 | 7.59 | 6.03 to 9.48 | 7.79 |
| Physics-guided | 6 | 693.15 | 650.90 | 6.09 | 4.59 to 7.67 | 6.14 |
| Physics-guided | 24 | 704.86 | 652.50 | 7.43 | 5.64 to 9.10 | 7.40 |

The research-question contrast is more selective after hourly history is controlled. All four procedures improve robustly at 1 hour. XGBoost and the physics-guided procedure also improve robustly at 3 and 6 hours. The remaining intraday cells are inconclusive, and no procedure establishes a robust 24-hour benefit.

Table 4.6: Controlled intra-hour contribution relative to `weather_history`. Positive gain favours `combined`; confidence intervals use 168-hour blocks.

| Model | h | RMSE history | RMSE combined | Gain (%) | 95% CI (%) | Scale gain (%) | Finding |
| --- | --- | --- | --- | --- | --- | --- | --- |
| XGBoost | 1 | 235.17 | 230.70 | 1.90 | 1.06 to 2.91 | 2.08 | Robust improvement |
| XGBoost | 3 | 352.35 | 344.49 | 2.23 | 1.12 to 3.32 | 2.32 | Robust improvement |
| XGBoost | 6 | 422.14 | 410.07 | 2.86 | 1.92 to 3.80 | 2.97 | Robust improvement |
| XGBoost | 24 | 440.12 | 435.82 | 0.98 | 0.03 to 2.10 | 0.98 | Inconclusive |
| LSTM | 1 | 159.48 | 155.25 | 2.65 | 1.45 to 4.00 | 2.83 | Robust improvement |
| LSTM | 3 | 305.26 | 302.65 | 0.85 | -0.22 to 1.99 | 0.99 | Inconclusive |
| LSTM | 6 | 378.67 | 375.28 | 0.89 | 0.01 to 1.83 | 0.92 | Inconclusive |
| LSTM | 24 | 411.98 | 414.86 | -0.70 | -1.32 to 0.06 | -0.60 | Inconclusive |
| CNN-BiLSTM | 1 | 159.74 | 155.58 | 2.60 | 1.55 to 3.81 | 2.74 | Robust improvement |
| CNN-BiLSTM | 3 | 304.50 | 300.92 | 1.18 | 0.08 to 2.04 | 1.21 | Inconclusive |
| CNN-BiLSTM | 6 | 378.61 | 373.57 | 1.33 | 0.34 to 2.16 | 1.27 | Inconclusive |
| CNN-BiLSTM | 24 | 390.54 | 399.77 | -2.37 | -4.10 to -0.48 | -2.25 | Inconclusive |
| Physics-guided | 1 | 159.55 | 155.53 | 2.52 | 1.31 to 3.78 | 2.68 | Robust improvement |
| Physics-guided | 3 | 305.24 | 300.28 | 1.63 | 0.57 to 2.52 | 1.69 | Robust improvement |
| Physics-guided | 6 | 379.18 | 369.17 | 2.64 | 1.64 to 3.55 | 2.60 | Robust improvement |
| Physics-guided | 24 | 391.49 | 400.81 | -2.38 | -4.14 to -0.41 | -2.23 | Inconclusive |

Positive intervals alone do not determine the label. XGBoost at 24 hours and CNN-BiLSTM at 3 and 6 hours have intervals above zero, but at least one corrected dependence-sensitive test fails. The negative 24-hour intervals for both CNN-BiLSTM variants also remain inconclusive after family-wise correction.

% [FIGURE: feature_ablation_rmse.png]

% [FIGURE: xgboost_feature_importance.png]

Ordinary hourly generation history is the larger effect, reducing meteorology-only RMSE by 41.03 to 77.07 percent, versus 1.26 to 11.46 percent for dispersion without history. Without history, every procedure benefits at every horizon. Once history is present, all four benefit robustly at 1 hour, only XGBoost and the physics-guided model do so at 3 and 6 hours, and none does at 24 hours. Neural 24-hour gains of 7.43 to 8.06 percent without history become changes of -0.70 to -2.38 percent with it, indicating redundancy. XGBoost settings are shared within each horizon and selected across all four arms, ruling out tuning on `combined` alone as the explanation.

## 4.5 Baseline Robustness, Weather Regimes and Stability

Across the 12 model-baseline comparisons for each learned procedure, XGBoost has 10 robust wins and each neural procedure has 11. All four learned procedures robustly outperform persistence at 1-6 hours but not at 24 hours, and all beat rolling climatology at every horizon. XGBoost does not robustly beat SARIMA at 1 hour; all neural procedures do.

The count of robust wins is a summary, not a league table: each 48-test bandwidth family includes references of different strength, and all horizons contribute equally.

% [FIGURE: all_models_observed_by_horizon.png]

Weather-regime analysis shows that the controlled effect is not uniform. Table 4.7 uses target-time clearness index, which was not supplied to the models. The unequal strata contain 578 overcast hours (576 at 24 hours), 2,008 variable (2,000) and 1,688 clear (1,683). With no within-regime multiplicity correction or cell-level intervals, these repeated views of one test year are descriptive; negative cells are retained.

% [FIGURE: clearness_index_regimes.png]

Table 4.7: Controlled intra-hour RMSE gain (%) by daylight weather regime, from the predeclared `weather_history` versus `combined` comparison.

| Horizon | Overcast | Variable | Clear | Range | Negative cells |
| --- | --- | --- | --- | --- | --- |
| 1h | +1.78 | +1.13 | +4.13 | -0.44 to +6.14 | 2 of 12 |
| 3h | +2.02 | +1.08 | +1.66 | -1.62 to +3.53 | 2 of 12 |
| 6h | -0.78 | +1.98 | +2.32 | -2.67 to +3.79 | 3 of 12 |
| 24h | +0.83 | -0.67 | -1.99 | -3.94 to +3.85 | 8 of 12 |

Each cell is the mean across the four learned procedures, so the table summarises direction rather than any single model. The regime analysis remains descriptive and outside the confirmatory families.

The pattern contradicts the predeclared regime hypothesis, which expected the largest gain under variable cloud. Clear conditions rank highest at 1 and 6 hours, while variable conditions rank highest at no horizon. At 24 hours both variable and clear means are negative.

A second reading fits better. Under broken cloud the hourly mean already reflects the cloud-driven variation within it, so dispersion adds little beyond the level it is computed from; under clear skies that level is largely set by solar geometry, so residual dispersion still signals departure from it. The feature is thus most useful where hourly history is least informative alone, the opposite of what was predicted. The design cannot separate that account from the alternatives below: clearness-index thresholds are arbitrary, eight-site averaging smooths the spatial variation driving real cloud transitions, and regime counts are unequal.

% [FIGURE: weather_regime_gain.png]

% [FIGURE: learned_model_weather_regimes.png]

Individual-seed RMSE confirms training variation. For combined inputs at 24 hours, standard deviation is 2.88 MW for LSTM, 14.91 for CNN-BiLSTM and 16.78 for the physics-guided version. Forecast averaging yields the headline values in Table 4.2, which are not the means of seed-level RMSE.

The matched pair has similar 24-hour dispersion, showing that the instability is not created solely by the penalty. Five seeds do not fully characterise training variation, and the paired test intervals do not include it.

The moving-block intervals and Diebold-Mariano tests treat each five-seed mean forecast as fixed. Their uncertainty covers dependence within 2025 and contains no training-variance component, so small neural differences remain conditional on these fitted ensembles.

% [FIGURE: neural_convergence.png]

The inverse-validation-RMSE ensemble is balanced rather than dominated by one component, yet it never supplies the best combined-input point estimate. Its RMSE is 161.39, 303.88, 373.77 and 407.40 MW, better than XGBoost but worse than the strongest neural model at each horizon.

Weights remain between 0.22 and 0.26, so the result is closer to variance smoothing than specialised regime switching.

The representative trajectories and scatter plots show where aggregate metrics originate. One-hour forecasts follow the daily envelope closely, while longer horizons smooth peaks and underpredict some high-output observations. Dense night-time points near zero improve all-hour scores, whereas daylight scatter widens with horizon. These figures are diagnostic examples, not independent tests.

The evidence hierarchy is consistent. Hourly generation history has the largest effect. Intra-hour summaries add smaller robust gains without history, but after controlling history their support concentrates at 1 hour, with 3 and 6-hour gains for XGBoost and the physics-guided procedure. Physical penalties improve only the 6-hour result robustly. Relative, peak-normalised and absolute RMSE agree, as does the output-scale sensitivity check. Conservative dependence assumptions turn marginal findings inconclusive rather than reversing robust claims, supporting a short-horizon feature conclusion while leaving day-ahead value open. Reporting only the best cell would obscure this pattern: LSTM with combined inputs leads at 1 hour, whereas unconstrained CNN-BiLSTM with history but no dispersion leads at 24 hours.

% [FIGURE: actual_vs_predicted_test.png]
