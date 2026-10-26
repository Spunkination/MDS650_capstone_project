# Chapter 5: Discussion

## 5.1 Answer to the Research Question

Scale-normalised five-minute intra-hour features improve short-horizon forecast skill beyond ordinary hourly history, but the result depends on model and horizon. In the controlled comparison, all four learned procedures gain robustly at one hour. XGBoost and the physics-guided CNN-BiLSTM also gain robustly at three and six hours, while the LSTM and unpenalised CNN-BiLSTM do not satisfy the full decision rule there. No procedure gains robustly at 24 hours. The robust controlled improvements range from 1.63 to 2.86 percent.

The complementary comparison without hourly history is broader but less selective. Adding the two dispersion summaries to meteorology improves all four learned procedures robustly at all four horizons, by 1.26 to 11.46 percent. The meteorological base is weak in absolute terms, including an XGBoost one-hour RMSE of 722.73 MW against persistence at 469.06 MW, because no feature set contains weather for the target period. Part of that gain therefore substitutes for forward weather an operational system would already hold. The controlled comparison is the direct answer because both arms contain the same hourly generation history.

Most of the value of AEMO data therefore comes from the latest hourly level and trailing summaries. Across the learned procedures, they reduce meteorology-only RMSE by 41.03 to 77.07 percent, compared with 1.26 to 11.46 percent for the dispersion columns alone. The five-minute summaries are incremental short-horizon signals, not substitutes for ordinary generation history.

The pattern suggests representation matters: tree splits can use a variability threshold directly, and the physics-guided sequence model benefits at each intraday horizon. The LSTM and unpenalised CNN-BiLSTM absorb a robust gain only at one hour under the stringent conjunction rule, possibly because their sequence representations already reconstruct part of the recent dynamics.

The 24-hour reversal is reasonable: variability in the latest completed hour can inform cloud motion over the next few hours but is unlikely to describe conditions a day later, while learned noise or regime-specific associations persist. Chapter 4 offers a second explanation at that horizon, namely that the recurrent fits stop improving after about nine to twelve epochs, so the day-ahead models extract less structure from the same window. This design does not separate the two accounts, so the recommendation to omit the features at 24 hours is precautionary rather than explained. Either way the results favour horizon-specific input policies over assuming that a richer feature vector is universally preferable.

## 5.2 The Regime Hypothesis Was Not Supported

The regime hypothesis predicted the largest intra-hour gain under variable cloud, since coarse hourly reanalysis should miss rapid change. The measured ordering does not support that prediction. Averaged across the four learned procedures, the one-hour gain is 4.13 percent under clear skies, 1.78 percent overcast and 1.13 percent variable. Clear skies also give the largest mean gain at six hours, while the 24-hour means are negative under clear and variable conditions.

This is a failed prediction, not a reversal. The regime breakdown is a secondary subgroup analysis with unequal strata of 578 overcast, 2,008 variable and 1,688 clear hours. It has no multiplicity correction or interval for an individual cell, and the test year had already been inspected. It should not outweigh the result it qualifies.

Two competing accounts fit the observed direction better, and both are testable. The first account is redundancy. Under broken cloud, the hourly mean already contains that hour's cloud-driven variation, so a dispersion summary from the same intervals adds little to the level it accompanies. Under clear skies, the hourly level is nearly predictable from solar geometry and calendar position, so residual dispersion still carries information about departures from that expectation. That account predicts the feature is most useful where hourly history is least informative.

The second is a market-clearing account. Both dispersion features are computed from `SS_SOLAR_CLEAREDMW`, the same market-cleared series as the target, which sits materially below unconstrained availability at midday. Taking midday as market-clock label hours 10:00 to 14:00 with unconstrained output above 50 MW gives 94,018 intervals. Cleared output is below 99 percent of available output in 75.6 percent of cases, and the mean cleared-to-available ratio is 0.891, falling from 0.909 in 2021 to 0.830 in 2025. Curtailment and dispatch therefore change the coefficient of variation and the relative range directly, so the clear-regime ordering could reflect when clearing departs from available output rather than when dispersion carries weather information. Because the unconstrained series `SS_SOLAR_UIGF` is present in the same export but excluded, re-deriving the two summaries from it, or stratifying by the cleared-to-available ratio, would separate the explanations. Neither test is reported here, so both accounts remain open.

Two design features limit this account. Eight-site equal-weight averaging smooths the spatial structure that produces real cloud transitions, so the variable regime may be less variable than labelled. The clearness-index thresholds at 0.3 and 0.7 are conventional, so moving them would reassign hours.

The main finding does not depend on the regime result, because the primary comparison was predeclared on the full sample and unaffected by later grouping; what changes is the interpretation, since within-hour dispersion is not most valuable where cloud is most active.

## 5.3 Model Interpretation

The neural models generally outperform XGBoost, though their ordering depends on horizon. LSTM leads at one hour, the physics-guided CNN-BiLSTM at three and six hours, and the unpenalised CNN-BiLSTM at 24 hours. Both CNN-BiLSTM variants robustly outperform XGBoost at every horizon; LSTM does so at 1, 3 and 6 hours. Most neural-to-neural gaps are inconclusive. The exceptions are the CNN-BiLSTM variants over LSTM at 24 hours and the physics-guided model over the unpenalised control at six hours.

The matched CNN-BiLSTM control isolates the two loss penalties while holding architecture, inputs and training protocol fixed. Their 1.18 percent six-hour gain is robust, but the comparisons at 1, 3 and 24 hours are inconclusive. The evidence therefore supports a horizon-specific penalty effect, not a general advantage or a physical mechanism.

The ensemble adds computation without a point-accuracy advantage on the present objective: balanced validation weights improve on XGBoost but do not beat the strongest neural forecast, and no failure-injection, missing-model or cross-year experiment tests operational resilience.

Model choice should therefore weigh evidence strength against maintenance cost. On combined inputs, the best RMSEs are 155.25 MW for LSTM at one hour, 300.28 and 369.17 MW for the physics-guided model at three and six hours, and 399.77 MW for the unpenalised CNN-BiLSTM at 24 hours. The small intraday neural gaps are mostly unresolved, while both CNN-BiLSTM variants have clearer day-ahead evidence than LSTM. The best day-ahead input set still excludes the intra-hour variables.

## 5.4 Baselines and Horizon Effects

Persistence is a demanding 24-hour reference because solar output is strongly periodic daily. The physics-guided combined model improves its RMSE by 5.38 percent, but the corrected comparison is not robust at the conservative dependence setting, where weekly serial dependence and family-wise multiplicity apply. At 1-6 hours persistence deteriorates as the target crosses the solar ramp, so SARIMA is the more informative short-horizon comparator: all three neural procedures robustly outperform it at every horizon, while XGBoost does not at one hour.

Skill is therefore reference-dependent. A 77.38 percent improvement over six-hour persistence sounds dramatic, but persistence is unsuitable across a solar-cycle transition and the forecast improves less against SARIMA. A small day-ahead persistence gain only means that copying the latest hour to the same clock time a day later captures much of the daily pattern.

## 5.5 Relation to Previous Research

The result is consistent with work emphasising complementary data sources and recent generation history [14], while narrowing the claim through a controlled ablation. Prior work on intra-hour irradiance variability motivates the feature design [4], but this project forecasts regional market-cleared generation rather than site irradiance, so its effect sizes and mechanism should not transfer.

Two published effect sizes frame the magnitudes: Perera et al. [18] report a 3.81 percent average MASE reduction from forecast combinations, and Abdullah and Mohammed [20] report GHI RMSE of 19.53 W/m2 against 30.64 W/m2 for attention-based alternatives. The robust controlled gains of 1.63 to 2.86 percent are of the same order as the first and smaller than the second, but targets, inputs and protocols differ, so the comparison is contextual, not a replication.

MERRA-2 supports a reproducible historical benchmark [7], but it is retrospective reanalysis, not an operational forecast available at issue time, so the study evaluates conditional skill with historical weather fields and does not quantify performance under live numerical weather prediction latency or error. The physics-guided literature [20] supports domain constraints, but the present loss is narrower than an irradiance-to-power physical model, and sequence-to-sequence methods [13] are not reproduced, since each horizon uses an independent scalar-output model.

The benchmark also differs from plant-level PV studies in scale. Regional averaging reduces plant noise, while NSW-wide clouds, capacity additions and dispatch constraints add non-stationarity. Market-cleared generation is not interchangeable with irradiance or AC power, so agreement with earlier rankings is convergence, not replication.

## 5.6 Strengths and Limitations

The main strengths are a chronological held-out year, exact elapsed-time alignment, prevention of gap-bridging sequences, conservative duplicate handling, common windows, five neural seeds, feature-neutral XGBoost tuning, a complete four-model factorial and dependence-sensitive inference. The matched CNN-BiLSTM control separates the loss penalties from the architecture.

Several limitations bound the conclusions. First, 2025 is one test year: moving-block intervals describe uncertainty within that year, not variation across future years or fleet growth, and contain no training-seed component. Second, the target excludes most rooftop PV, which nationally supplied 13.9 percent of electricity generation in 2025 against 7.7 percent for utility-scale solar [37], so the segment outside this benchmark is the larger one; the target also includes market interventions and curtailment. Third, eight-site mean reanalysis cannot resolve all spatial cloud motion. The locations are population centres spanning the coast and inland slopes, chosen as a coast-to-inland spread of regional weather proxies with no capacity weighting and no formal site-selection procedure. Inland generation may therefore be under-represented. Fourth, operational feature availability, ingestion delay and live forecast quality are untested. Fifth, XGBoost uses one fitted model per block while neural forecasts average five fits, so between-procedure point estimates do not share the same seed averaging. Sixth, clipping at zero affects a non-trivial share of outputs, and the regime and feature-importance analyses are descriptive rather than causal. Finally, the tested five-minute representation contains only two summaries of the latest completed hour, not ramp timing, interval order or the raw twelve values.

Inference remains conditional on several choices: moving blocks approximate serial dependence but cannot reproduce every seasonal or regime transition, and Bonferroni correction is conservative and reduces power. The intersection rule is more demanding, because one unstable diagnostic changes the label to inconclusive, which explains cases such as the LSTM six-hour gain.

The input control is stronger than a broad ablation but is not a randomised causal intervention. Adding two columns can change optimisation paths, tree splits and learned regularisation. The reported gain therefore belongs to the complete training procedure with or without those columns, not to the energy value of variability.

The duplicate policy is the least assumptive choice for the legacy archive: keeping the first or last row would make the target depend on file ordering, and averaging conflicting rows could create a value matching neither dispatch scenario. Dropping a conflicting group loses data but keeps the target interpretable; no excluded hour occurs in the test year, so the direct test metrics are unaffected, though the training sample can still shift through the 120 excluded hourly bins.

## 5.7 Implications

For this NSW1 task, recent hourly generation should be a core operational input. The two intra-hour summaries are candidates at short horizons, where robust controlled gains range from 1.63 to 2.86 percent. They are robust for XGBoost and the physics-guided model at 1, 3 and 6 hours, and for the LSTM and unpenalised CNN-BiLSTM at one hour only. Their value is conditional on reliable five-minute telemetry, and no cost side is estimated here, so the ingestion decision remains open. At one hour XGBoost remains much weaker at 230.70 MW than LSTM at 155.25 MW despite its feature gain. At 24 hours the features should not be assumed beneficial.

No architecture dominates every horizon. LSTM is strongest at one hour, the physics-guided model at three and six hours, and the unpenalised CNN-BiLSTM at 24 hours. Deployment would still require issue-time weather, latency monitoring, missing-data fallback, capacity-growth recalibration and out-of-year validation, and RMSE cannot be converted to trading value without prices, exposure and a decision model.

A staged deployment study would begin with shadow forecasts timestamping each prediction and recording every feature's publication time, then monitor performance by horizon, daylight condition, output level and missing-data state.

Capacity growth and interpretability stability also deserve work, and computational reproducibility should not be confused with numerical identity.
