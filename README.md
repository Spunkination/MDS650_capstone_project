# Short-Term Solar PV Forecasting in NSW Using Machine Learning

### Scale-Normalised Intra-Hour Variability Features Under Controlled History

Kingsley Ogadinma Onyekwele
Master of Data Science capstone project (SPI240350)
Sydney Polytechnic Institute · Supervisor: Dr Anh Nguyen · September 2026

---

## Abstract

This project forecasts solar photovoltaic generation for the Australian Energy
Market Operator's NSW1 region at horizons from five minutes to one day, and
asks whether scale-normalised measures of five-minute intra-hour variability
add forecast skill once ordinary hourly generation history is held constant.
The analysis merges five-minute AEMO dispatch data for 2020–2025 with hourly
MERRA-2 reanalysis meteorology for eight NSW sites, and evaluates four feature
sets with XGBoost, a unidirectional LSTM, a CNN-BiLSTM in both unpenalised and
physics-guided forms, and a weighted ensemble, against persistence, rolling
climatology and SARIMA baselines. Under control the intra-hour features add
1.90 to 2.65 percent at one hour across all four procedures, less at three and
six hours (XGBoost and the physics-guided model only), and nothing that survives
the dependence tests day-ahead. Hourly
history is the larger effect: it reduces meteorology-only RMSE by 41.03 to
77.07 percent, against controlled gains of 1.63 to 2.86 percent. Most of the
intra-hour signal is therefore already carried by the history it is compared
against. Day-ahead is where forecasting fails: all four learned models are
strongly negatively biased at 24 hours, and none robustly beats persistence.

---

## Research question

To what extent do scale-normalised AEMO five-minute intra-hour variability
features improve forecast skill after controlling for ordinary hourly
generation history?

The question carries three aims. The first is to isolate the contribution of
the intra-hour features by holding hourly history fixed, rather than by
comparing feature sets that differ in several respects at once. The second is
to establish whether learned models beat the three baselines (persistence,
rolling climatology and SARIMA) at each horizon. The third is to characterise
where skill fails and why, with day-ahead performance treated as a result in
its own right rather than as an inconvenience.

---

## Data

| Source | Provider | Content | Period |
|---|---|---|---|
| MMSDM five-minute dispatch, NSW1 | AEMO, via NEMOSIS | Semi-scheduled solar cleared MW, resampled to hourly; target `SS_SOLAR_CLEAREDMW_mean` | 2020–2025 |
| MERRA-2 reanalysis | renewables.ninja | Hourly GHI, temperature and wind for eight NSW sites (Sydney, Newcastle, Wollongong, Central Coast, Tamworth, Bathurst, Orange, Dubbo), aggregated to an NSW-wide mean | 2020-09 to 2025 |

Raw source files are not redistributed in this repository, but they are
reproducible from the collection code in `code/data_collection/`.
`collect_aemo.py` pulls NSW1 dispatch data through NEMOSIS; `collect_data.py`
and `renewables_ninja_api.py` pull the MERRA-2 meteorology from
renewables.ninja, authenticated with `RENEWABLES_NINJA_API_KEY` in the
environment. The scripts write to `data/raw/`, which is gitignored and is where
the notebook expects to find `aemo_dispatchregionsum_nsw1.csv`,
`renewables_ninja_sydney.csv` and one `renewables_ninja_{site}.csv` per
additional site.

---

## Method

Four feature sets are defined and carried through every model: `weather`,
`weather_history`, `weather_intra` and `combined`, containing 11, 14, 13 and 16
variables respectively. The primary comparison is `weather_history` against
`combined`, because that pair holds hourly generation history constant while
adding only two quantities, the coefficient of variation and the relative
five-minute range, both scale-normalised so that they describe intra-hour shape
rather than output level.

Timestamps follow AEMO's fixed AEST. Exact-value duplicate rows are collapsed
and timestamps carrying conflicting target values are dropped, after which the
affected hourly bins fail a twelve-interval completeness rule and are excluded.
Explicit nighttime zeros are retained and incomplete source hours are excluded.
Issue times follow completion of the latest left-labelled hourly interval, and
every horizon assertion uses elapsed target-interval end times. Splits are
chronological, with 2025 held out for testing.

The learned-model design contains four configurations: XGBoost on the CPU, a
unidirectional LSTM, and a CNN-BiLSTM in both its unpenalised and
physics-guided forms on an accelerated PyTorch device. All four run the full
four-set ablation, so both intra-hour contrasts cover every learned procedure.
Neural headline forecasts are five-seed means and XGBoost is a single fitted
model. XGBoost is tuned once per horizon on `combined` and its selected
settings are reused across all four feature sets, which fixes the tuning budget
across the ablation.

The three baselines are persistence, rolling climatology using up to 365 past
observations from the target's local hour, and SARIMA fitted on a final
pre-test year with filtered states updated at each origin. Evaluation applies
twelve Bonferroni-corrected learned-model/horizon tests and dependence
sensitivities at 24, 48 and 168 hours. A gain counts as robust only where the
moving-block interval excludes zero, the corrected Diebold-Mariano test agrees
and the output-scale sensitivity check agrees, and pairwise ranking is reported
only under that same agreement. Daylight error, bias and weather-regime
diagnostics provide secondary interpretive context, and raw-MW findings are
checked against a past-only trailing-output-scale sensitivity.

---

## Findings

On combined inputs the LSTM gives the lowest one-hour error, at 155.25 MW RMSE.
The physics-guided CNN-BiLSTM leads at three hours (300.28 MW) and six hours
(369.17 MW), and the unpenalised CNN-BiLSTM at 24 hours (399.77 MW). The
24-hour minimum over all feature sets is 390.54 MW, from the same model on
`weather_history` without the intra-hour variables.

The controlled comparison is the primary result. Adding the two
scale-normalised intra-hour features to `weather_history` gives a robust gain
at one hour for all four procedures (1.90 to 2.65 percent) and a gain that
holds at three and six hours for XGBoost and the physics-guided model, with no
gain at 24 hours for any model. The 24-hour point estimates are negative for
both CNN-BiLSTM variants and the LSTM.

Hourly history is the larger effect, reducing meteorology-only RMSE by 41.03 to
77.07 percent across the learned procedures, where the controlled gains range
from 1.63 to 2.86 percent. Measured without history, the dispersion columns are
worth 1.26 to 11.46 percent against a meteorology-only baseline, so most of
their information is already carried by the history they are compared against.

Against the three baselines, all three neural procedures win 11 of 12
model-horizon-baseline comparisons and XGBoost wins 10 of 12. No learned model
robustly beats persistence at 24 hours. All four beat rolling climatology at
every horizon. XGBoost does not robustly beat SARIMA at one hour; the three
neural procedures do.

The matched CNN-BiLSTM pair separates the loss penalties from the architecture.
Holding inputs, optimiser, seeds and early-stopping rule fixed, the
physics-guided version gains a robust 1.18 percent at six hours; the
comparisons at one, three and 24 hours are inconclusive.

Day-ahead is the weak point. All four learned models are strongly negatively
biased at 24 hours, from -93.88 to -132.44 MW, and neural early stopping
triggers within about 9 to 12 epochs, against 51 to 63 at one hour. The
weighted ensemble never supplies the best combined-input point estimate at any
horizon.

---

## Scope and limitations

Three constraints are built into the design.

The fixed-hyperparameter intervention fixes the tuning budget, but it may
favour the feature set used for selection, so the four sets are not
independently optimised and the ablation should not be read as a comparison of
tuned alternatives. Away from the matched pair, comparisons change
architecture, training dynamics or seed treatment together, so the pairwise
results describe whole procedures rather than the effect of any one component.
The 2025 holdout was inspected repeatedly during development, so the reported
tests are exploratory within that year rather than confirmatory; the dependence
sensitivities and the agreement rule applied to pairwise rankings are the
safeguards against reading noise as signal.

Two further limits apply. Combined-input XGBoost gain importance is aggregated
across the 24 lag positions and is descriptive, not causal. The weighted
ensemble is evaluated but never supplies the best combined-input point estimate
at any horizon, so it is reported as a negative result rather than as a
component of the recommendation. The findings concern 2025 market-cleared
utility-scale generation under retrospective reanalysis; they do not extend to
rooftop PV, unconstrained production or market value.

---

## Reproducing the analysis

```bash
python -m pip install -r requirements.txt
jupyter nbconvert --to notebook --execute --inplace analysis/solar_pv.ipynb
```

Python 3.12, PyTorch 2.5.1, XGBoost 3.0.1, pandas 2.3.3, numpy 2.1.3. The last
reportable run used Apple Silicon MPS for the neural tiers and the CPU for
XGBoost; the notebook's saved provenance block records the machine that
produced a given run.

The notebook is self-contained: it loads the data, merges the sources, runs the
training-only EDA, engineers the four feature sets, produces the chronological
splits, and trains and evaluates every model. A reportable run must start from
a fresh kernel, which the first code cell enforces. For a faster structural
check before committing to a reportable run:

```bash
SOLAR_RUN_PROFILE=development jupyter nbconvert --to notebook --execute --inplace analysis/solar_pv.ipynb
```

All data is held in memory and no intermediate artefact files are written. The
`final` profile requires an accelerated PyTorch device unless
`SOLAR_ALLOW_FINAL_CPU=1` is set explicitly.

---

## Structure of the submission

```
analysis/solar_pv.ipynb    # the complete analysis: load, merge, EDA, feature
                           #   engineering, splits, training, evaluation
manuscript/
  ch1_introduction.md      # the chapters in readable form, for browsing on
  ch2_literature.md        #   GitHub; the LaTeX tree is what typesets the PDF
  ch3_methodology.md
  ch4_results.md
  ch5_discussion.md
  ch6_conclusion.md
  latex/                   # the typesetting source, incl. compiled main.pdf
    figures/               # the figures as included by the LaTeX tree
  submitted/
    final_report.pdf       # the manuscript as submitted
    final_report.docx      # Word companion generated from the same sources
code/data_collection/      # AEMO (NEMOSIS) and renewables.ninja collection
requirements.txt
```

The PDF and Word files under `manuscript/submitted/` are the submitted
manuscript. Their typesetting source is `manuscript/latex/`, which is what
produces the PDF. The chapter Markdown files are a readable copy of the same
text, kept so the manuscript can be read in the browser without downloading the
PDF; they do not feed the build, so edit the LaTeX. The exports are committed
as rendered artefacts and the build tooling is not part of this repository.

---

## Citation

```bibtex
@misc{onyekwele2026solarpv,
  author       = {Onyekwele, Kingsley},
  title        = {Short-Term Solar PV Forecasting in {NSW} Using Machine
                  Learning: Scale-Normalised Intra-Hour Variability Features
                  Under Controlled History},
  year         = {2026},
  howpublished = {\url{https://github.com/Spunkination/MDS650_capstone_project}},
  note         = {Master of Data Science capstone project}
}
```

---

## Declaration and licence

The report text, figures and notebook are original work. Third-party
dependencies retain their own licences. AEMO and renewables.ninja source data
are not redistributed and remain subject to their providers' terms.

Released under the MIT Licence; see [LICENSE](LICENSE).