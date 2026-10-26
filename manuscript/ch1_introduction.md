# Chapter 1: Introduction

## 1.1 Background and Research Context

New South Wales (NSW) provides a regional setting for short-term solar photovoltaic (PV) forecasting. Variation in solar output complicates the balance between generation and demand, and forecasts can support scheduling and dispatch by anticipating it [5]. The target studied here is market-cleared generation: the semi-scheduled solar output AEMO accepts into dispatch after network constraints and price effects, which is not the same as available production. This project examines predictive accuracy; it does not estimate avoided curtailment, balancing costs or financial benefits.

Forecast value depends on lead time and the decision it supports, so the study evaluates one-hour, three-hour, six-hour and day-ahead horizons covering intraday and next-day planning. These are experimental forecasting horizons, not simulations of AEMO scheduling or settlement, and regional MW errors alone cannot establish participant-level costs or savings.

Relevant work includes hierarchical temporal convolutional networks for day-ahead regional forecasting in Western Australia [1], transformer models for irradiance and PV power forecasting [2], [3], intra-hour solar radiation variability [4], and satellite-based prediction at four Australian locations [5]. None supplies the matching NSW1 benchmark sought here. NSW exposure to East Coast Low events [6] provides regional context, although this project does not evaluate event-specific errors.

## 1.2 Research Problem and Motivation

The reviewed literature identifies no NSW benchmark combining local meteorological reanalysis with grid data for short-term solar PV generation forecasting. Prior Australian work has forecast at regional or state level from weather data and historical generation, and utility-scale forecasts feed national balancing through AEMO's operational processes, yet the retained review found no study sharing the NSW1 target, source combination and evaluation design. Section 2.1 records that those search notes lack a complete auditable screening ledger, so the gap is bounded by the retained material rather than established against the field. The nearest retained work is Perera et al. [1], who forecast distributed rooftop generation in Western Australia, and Prasad and Kay [5], who convert satellite-derived irradiance forecasts into power at Queensland and Victorian sites; neither combines NSW1 market-cleared generation with completed-hour AEMO history.

A second constraint is the meteorological source: MERRA-2 is retrospective reanalysis, not an operational forecast [7]. The experiment uses weather information associated with the forecast origin and its history, not archived numerical weather predictions issued in real time, so its value lies in comparing input sets and configurations within a retrospective benchmark. Neither a fixed deployment-error penalty nor unchanged operational model rankings can be inferred.

## 1.3 Research Questions and Objectives

The research question guiding this project is: to what extent do AEMO five-minute intra-hour features improve short-term solar PV forecast skill over meteorological inputs alone in New South Wales?

Three objectives structure the work. The first benchmarks XGBoost, a unidirectional LSTM, an unpenalised CNN-BiLSTM and a physics-guided CNN-BiLSTM against persistence, rolling climatology and seasonal ARIMA (SARIMA) over 1 h, 3 h, 6 h and 24 h horizons. The second estimates the incremental contribution of two scale-normalised intra-hour variables over meteorological inputs alone, and tests whether it survives once ordinary hourly generation history is present. The third compares the learned procedures and describes whether a validation-weighted ensemble improves their point accuracy. Each uses the same 2025 evaluation year and horizon-specific common forecast windows.

The primary hypothesis is that adding the coefficient of variation and relative five-minute range to a meteorology-only input set reduces RMSE at short horizons. Every learned procedure is fitted on all four feature sets, so the meteorology-only and history-controlled contrasts are estimated across the complete factorial. A gain counts as robust only when the 168-hour moving-block interval excludes zero, Diebold-Mariano tests remain significant after correction under all three dependence bandwidths, and an output-scale-relative sensitivity check agrees in direction. The same factorial estimates the larger contribution of ordinary hourly history.

The architecture comparison includes a matched control: the two CNN-BiLSTM procedures share architecture and training, differing only in the two loss penalties. The controlled contrast directly answers whether five-minute summaries add information beyond ordinary hourly history, while the meteorology-only contrast shows their complementary value without that history. The ensemble and regime analyses remain secondary.

The dataset combines hourly renewables.ninja meteorological reanalysis [7], [8] with 5-minute AEMO NEMOSIS dispatch data [9] for September 2020 to 2025, engineered into four information sets of up to 16 variables and split chronologically into training (2020 to 2023), validation (2024) and test (2025). After joining, 46,570 complete hourly observations remain, falling to 46,558 once rolling-feature warm-up is removed and before the contiguous-window burn-in described in Section 3.1.

## 1.4 Significance and Expected Contribution

The contribution is ranked. The primary contribution is an attribution of short-horizon forecast skill to AEMO five-minute intra-hour features, measured over meteorological inputs alone and then re-measured once ordinary hourly generation history is present. The supporting contribution is the applied NSW benchmark that makes that attribution possible: a four-set comparison of established architectures on market-cleared NSW1 generation, which the retained literature does not supply. Regime-conditioned errors describe where gains occur but remain exploratory. The novelty claimed is limited to that benchmark and comparison design, not a new forecasting architecture.

For AEMO and practitioners, the benchmark retrospectively compares input choices and established configurations. Its target is market-cleared NSW1 dispatch, excluding most rooftop production; operational application would require archived issue-time weather forecasts, latency checks and a missing-data policy, while extending it to distributed PV would require distributed observations.

The evaluation distinguishes the value of all generation-history predictors from the incremental contribution of their within-hour dispersion.

This distinction matters for both scientific interpretation and system design. If a model improves after all AEMO history is added, the change may simply reflect access to the latest generation level, an unusually strong predictor of near-term output, and not the five-minute sampling frequency. Conversely, a comparison of otherwise identical inputs differing only in intra-hour dispersion can estimate whether variability within the latest completed hour carries information its mean and trailing summaries omit. The two contrasts therefore answer different halves of the research question, and both are reported.

The four forecast horizons represent distinct decisions rather than interchangeable repetitions. One and three hours emphasise near-term cloud and ramp persistence, six hours crosses a substantial part of the daylight cycle, and 24 hours tests day-ahead periodicity.

The study also separates retrospective validity from operational readiness. A chronologically held-out year meaningfully tests temporal generalisation, but reanalysis weather, archived market data and zero assumed delivery delay do not recreate a live forecast service. The report therefore uses terms such as benchmark, association and conditional skill and avoids claiming deployable accuracy, economic savings or causal effects of weather regimes.

## 1.5 Thesis Outline

Chapter 2 synthesises the literature and research gap. Chapter 3 describes the data, models and evaluation design. Chapter 4 presents the results, Chapter 5 interprets them and their limitations, and Chapter 6 states the contribution and future work.

% [FIGURE: thesis_progression.png]
