# Chapter 6: Conclusion

## 6.1 Summary

This study asked to what extent AEMO five-minute intra-hour features improve short-term solar PV forecast skill over meteorological inputs alone in New South Wales. The controlled answer is positive at short horizons and disappears at 24 hours. All four learned procedures gain robustly at one hour; XGBoost and the physics-guided CNN-BiLSTM also gain at three and six hours, while the LSTM and unpenalised CNN-BiLSTM do not satisfy the full decision rule there. No procedure gains robustly at 24 hours.

Robust controlled gains range from 1.63 to 2.86 percent, while ordinary hourly history supplies the dominant reduction. In the complementary comparison without history, all four procedures improve robustly at every horizon by 1.26 to 11.46 percent. Throughout this report, robust means agreement across interval, Diebold-Mariano and output-scale checks within the 2025 test year, and the intervals contain no training-seed component.

On combined inputs, LSTM records the lowest one-hour RMSE at 155.25 MW, the physics-guided model leads at three and six hours with 300.28 and 369.17 MW, and the unpenalised CNN-BiLSTM leads at 24 hours with 399.77 MW. Its `weather_history` result of 390.54 MW is the lowest day-ahead feature-set value, showing that the intra-hour addition is not automatically helpful. The matched CNN-BiLSTM comparison attributes a robust 1.18 percent gain to the loss penalties only at six hours. The ensemble improves on XGBoost but not on the best neural forecast. These findings concern 2025 market-cleared utility-scale generation under retrospective reanalysis, not rooftop PV, unconstrained production or economic value.

## 6.2 Contributions

The primary contribution is an answer to the research question: AEMO five-minute intra-hour features carry a modest short-horizon increment beyond ordinary hourly history, with no robust day-ahead benefit. The evidence base is a self-contained regional benchmark joining AEMO five-minute generation with hourly weather, a duplicate and alignment policy, and a four-set feature design. Four learned procedures and three baselines are evaluated on common windows, with five averaged seeds for each neural procedure. A controlled result counts as robust only when block-bootstrap intervals, multiple Diebold-Mariano bandwidths and an alternative output scale agree. The notebook completed from a fresh kernel on Apple Silicon using MPS.

The evaluation controls also transfer: exact elapsed-time joins stop missing observations from silently redefining a horizon, contiguous sequences stop a recurrent model from treating gaps as adjacent hours, and multiple dependence bandwidths reveal when a significant result relies on a short-memory assumption.

Two findings extend beyond that answer. First, the complete factorial design separates the frequency of a data source from its incremental information across all four procedures. Without hourly history, gains range from 1.26 to 11.46 percent and are robust in all 16 cells. After controlling for history, the robust gains narrow to 1.63 to 2.86 percent and occur in eight of 16 cells. The increment shrinks as a completed hour's mean becomes available, which argues for controlled ablations over single-baseline comparisons.

Second, validation error bottomed out after roughly sixty epochs at one hour but within about nine to twelve epochs at day-ahead range. The recurrent models therefore extracted far less structure from an identical input window as the horizon lengthened, reframing a model-ranking result as a training-behaviour result.

## 6.3 Recommendations and Future Work

An operational extension should replace reanalysis with archived issue-time weather forecasts, reproduce the comparison across held-out years and other NEM regions, and define latency, record conflicts, missing-feature fallbacks, capacity-change monitoring and retraining triggers.

Further experiments should vary the penalty weights and constraints within the matched CNN-BiLSTM design, and test whether richer intra-hour representations add value beyond the two summaries. Probabilistic forecasts should be assessed with calibration, coverage and sharpness rather than point RMSE alone, and regime and ramp-event evaluation predeclared and powered. When additional years become available, rolling-origin evaluation would show whether the controlled feature effect persists through different weather years and fleet growth, while cross-region evaluation could separate NSW-specific conclusions from general properties of utility-scale solar forecasting.

Operational research should preserve the same data rules: auditable quarantine for conflicting records, deterministic de-duplication of identical re-deliveries, a documented fallback rather than silent zero filling, and publication timestamps instead of assumed immediate availability. Forecast value should also be measured in the decision context, since an operator may prioritise ramp warnings while a participant faces imbalance costs.

The recommended sequence is evidence-led: preserve hourly generation history and the strong baselines, add the two scale-normalised intra-hour summaries where each deployed model has a robust controlled gain, and omit them at 24 hours. Architecture should be chosen under measured latency and maintenance constraints rather than treating small RMSE gaps as decisive. Only then should a richer ensemble or high-frequency representation follow. The next validation should predeclare what would change this conclusion: retain the intra-hour columns only where the controlled gain remains positive with a 168-hour interval excluding zero and the corrected Diebold-Mariano and output-scale checks agree. Latency should be tested on archived issue-time weather or audited AEMO publication timestamps, and regime and ramp-event tests should state their required sample size before the strata are inspected.

This sequence protects the main conclusion from deployment-architecture changes: even if a future transformer, probabilistic model or improved weather source lowers absolute error, the controlled question can be repeated: does five-minute variability add information beyond the latest hourly level and trailing history?

## 6.4 Closing Statement

Recent hourly generation is the central predictive input in this benchmark. Five-minute variability adds smaller but repeatable short-horizon information over meteorological inputs alone, and its day-ahead value is unsupported, favouring controlled feature integration over claims that higher-frequency data or greater model complexity must improve every horizon. The tested object is two scale-normalised scalars computed from the twelve five-minute values of the latest completed hour; ramp timing, within-hour shape and the raw twelve values were not tested, so the finding concerns this representation, not five-minute sampling in general.
