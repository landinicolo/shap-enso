# Literature Review: ML-Based ENSO Prediction and XAI Explainability

*Compiled June 2026. Covers 2018–2025.*

---

## Key Finding: SHAP Is a Genuine Gap

The field has converged on **Integrated Gradients (IG)** and **Layer-wise Relevance Propagation (LRP)** as the dominant XAI tools for ENSO prediction models. SHAP has not been applied prominently to this problem — applying TreeExplainer (for XGBoost), GradientExplainer (for LSTM), and DeepExplainer (for CNN) systematically across lead times, seasons, and ENSO phases represents a clear novelty contribution.

---

## 1. ML Models for ENSO Prediction

### Ham et al. 2019 — The Foundational Paper
**Deep learning for multi-year ENSO forecasts**
- **Authors:** Ham, Y.-G., Kim, J.-H., and Luo, J.-J.
- **Journal:** Nature, vol. 573, pp. 568–572
- **DOI:** [10.1038/s41586-019-1559-7](https://doi.org/10.1038/s41586-019-1559-7)
- **Code:** https://github.com/jeonghwan723/DL_ENSO

A CNN pretrained on 21 CMIP5 model simulations and fine-tuned on SODA reanalysis (1871–1973) achieves Niño 3.4 correlation skill >0.5 up to 17-month lead, outperforming dynamical models in the 1984–2017 validation window. Established the **CMIP pretraining → reanalysis fine-tuning** paradigm that nearly all subsequent work follows.

---

### ResoNet — Mu et al. 2024 (Closest Comparator to This Project)
**ResoNet: Robust and Explainable ENSO Forecasts with Hybrid Convolution and Transformer Networks**
- **Authors:** Mu, B. et al.
- **Journal:** Advances in Atmospheric Sciences, vol. 41, no. 7, pp. 1289–1298
- **DOI:** [10.1007/s00376-024-3316-6](https://doi.org/10.1007/s00376-024-3316-6)
- **arXiv:** [2312.10429](https://arxiv.org/abs/2312.10429)

CNN backbone + transformer; 19-month lead skill. Uses **Integrated Gradients** to recover the Recharge Oscillator paradigm, the Seasonal Footprinting Mechanism, and El Niño/La Niña asymmetry in attribution space. This is the most direct comparator: the present project aims to do what ResoNet did with IG, but with SHAP, and with an explicit spring-barrier and asymmetry decomposition.

---

### CTEFNet — 2025
**Toward long-range ENSO prediction with an explainable deep learning model**
- **Authors:** Chen, Q. et al.
- **Journal:** npj Climate and Atmospheric Science, vol. 8, article 259
- **DOI:** [10.1038/s41612-025-01159-w](https://doi.org/10.1038/s41612-025-01159-w)
- **arXiv:** [2503.19502](https://arxiv.org/abs/2503.19502)

CNN-transformer hybrid trained on 18 CMIP6 models, validated on ORAS5/ERA5. Achieves 20-month effective lead time and partially mitigates the spring predictability barrier. Uses gradient-based sensitivity analysis as XAI. Gradient sensitivity reveals that spring-initialized forecasts gain skill from Indian Ocean and Atlantic precursors.

---

### 3D-STransformer — Lian et al. 2025
**A Deep Learning-Based Long-Term ENSO Forecasting Model: 3D-STransformer**
- **Journal:** Journal of Geophysical Research: Machine Learning and Computation
- **DOI:** [10.1029/2024JH000412](https://doi.org/10.1029/2024JH000412)

Spatio-temporal transformer trained on 23 CMIP6 models (1850–2014) with SODA reanalysis transfer training. Maintains high Niño 3.4 skill up to 20 months. Multi-head attention over 3D ocean state captures ENSO dynamics inaccessible to 2D CNN models.

---

### Lguensat et al. 2022
**Deep learning for skillful long-lead ENSO forecasts**
- **Journal:** Frontiers in Climate
- **DOI:** [10.3389/fclim.2022.1058677](https://doi.org/10.3389/fclim.2022.1058677)

CNN trained on CMIP6 + ORAS5/ERA5. Demonstrates that adding subsurface heat content (0–300 m, a proxy for thermocline depth / D20) substantially improves skill at leads >12 months. Motivates including the 20°C isotherm depth as a predictor.

---

## 2. XAI Methods Applied to ENSO

### Toms et al. 2020 — Foundational XAI-Climate Paper
**Physically Interpretable Neural Networks for the Geosciences: Applications to Earth System Variability**
- **Authors:** Toms, B.A., Barnes, E.A., and Ebert-Uphoff, I.
- **Journal:** Journal of Advances in Modeling Earth Systems (JAMES), vol. 12, no. 9
- **DOI:** [10.1029/2019MS002002](https://doi.org/10.1029/2019MS002002)
- **arXiv:** [1912.01752](https://arxiv.org/abs/1912.01752)

Introduces **Layer-wise Relevance Propagation (LRP)** to the geoscience community and applies it to ENSO phase identification and seasonal surface temperature prediction. LRP decomposes network decisions back onto input pixels, generating physically interpretable heatmaps that identify known ENSO precursor patterns. The first systematic demonstration that neural network attribution can recover physical mechanisms rather than just prediction skill.

---

### Shin et al. 2022
**Application of Deep Learning to Understanding ENSO Dynamics**
- **Journal:** Artificial Intelligence for the Earth Systems (AIES), vol. 1, no. 4
- **DOI:** [10.1175/AIES-D-21-0011.1](https://doi.org/10.1175/AIES-D-21-0011.1)

Applies CNN to CMIP6 ENSO prediction and introduces "contribution maps" (gradient-based) and "contribution sensitivity" (perturbation-based) to determine which grid cells and variables drive predictions. Explicitly connects attribution outputs to known physical mechanisms: thermocline depth and zonal wind stress are the dominant contributors at longer lead times.

---

### Rivera Tello et al. 2023
**Explained predictions of strong eastern Pacific El Niño events using deep learning**
- **Journal:** Scientific Reports, vol. 13
- **DOI:** [10.1038/s41598-023-45739-3](https://doi.org/10.1038/s41598-023-45739-3)

Builds a deep learning model specialized for forecasting strong EP El Niño onset. Uses LRP relevance maps to trace precursors of the 2023 strong El Niño. Demonstrates ENSO-type asymmetry in attribution: strong EP events have distinct precursor signatures from CP events, with key signals in the western/central Pacific.

---

### Nearing et al. 2023
**Explainable deep learning for insights in El Niño and river flows**
- **Journal:** Nature Communications, vol. 14, article 339
- **DOI:** [10.1038/s41467-023-35968-5](https://doi.org/10.1038/s41467-023-35968-5)
- **arXiv:** [2201.02596](https://arxiv.org/abs/2201.02596)

Applies saliency-map XAI to CNNs predicting river flows from global SST. Reveals SST information regions beyond ENSO indices, including inter-basin teleconnections. Demonstrates that XAI extracts predictive insight not captured by standard Niño indices — motivating SHAP analysis to discover new physical information beyond known precursors.

---

### XAI + WNP Precursor — 2024
**Explainable AI in lengthening ENSO prediction from western North Pacific precursor**
- **Journal:** Ocean Modelling
- **DOI:** [10.1016/j.ocemod.2024.001185](https://doi.org/10.1016/j.ocemod.2024.001185)

XAI identifies western North Pacific (WNP) SST anomalies as an emerging ENSO precursor not captured by standard basin indices. Integrating WNP SSTA with XAI-guided feature selection raises 1-year-ahead classification accuracy from 60% to >85%. Shows that XAI can both identify physical precursors and operationally improve forecast skill.

---

## 3. Spring Predictability Barrier in ML Models

### DeepConvLSTM — 2020
**Prediction of ENSO Beyond Spring Predictability Barrier Using Deep Convolutional LSTM Networks**
- **Journal:** IEEE Geoscience and Remote Sensing Letters
- **DOI:** [10.1109/LGRS.2020.3032353](https://doi.org/10.1109/LGRS.2020.3032353)

Demonstrates that deep ConvLSTM networks can extend skillful ENSO predictions beyond the spring barrier compared to linear statistical models, attributing this to the model's ability to capture nonlinear relationships in multi-variable inputs (SST, heat content, wind stress) through boreal spring.

---

### GL-Geoformer — Science Advances 2025
**Tropical basin interactions reduce spring predictability barrier of ENSO in a deep learning model**
- **Journal:** Science Advances
- **DOI:** [10.1126/sciadv.aeb0901](https://doi.org/10.1126/sciadv.aeb0901)

A transformer-based model achieves skillful ENSO predictions up to 16 months when initialized in spring by explicitly modeling 3D temperature and wind anomalies across all three tropical basins (Indian, Atlantic, Pacific). Shows that the spring barrier in ML models is reducible by incorporating inter-basin teleconnections — a physically grounded finding with direct implications for what SHAP maps should show around MAM initialization.

---

### PNAS 2025
**Identifying key convection-sensitive oceanic regions to weaken the ENSO spring predictability barrier**
- **Journal:** Proceedings of the National Academy of Sciences
- **DOI:** [10.1073/pnas.2512725123](https://doi.org/10.1073/pnas.2512725123)

Uses a dynamical model sensitivity framework to identify which oceanic regions, when better observed, most reduce the SPB. Provides a physical baseline: if SHAP analysis on an ML-ENSO model identifies the same oceanic regions as critical for spring-initialized forecasts, it validates the physical fidelity of the attribution.

---

## 4. ENSO Asymmetry (El Niño vs. La Niña) in ML and XAI

**ResoNet (Mu et al. 2024)** — IG attribution at 18-month lead shows El Niño development linked to southern Pacific SST precursors, while La Niña is tied to eastern equatorial Pacific signals. First deep-learning quantification of El Niño/La Niña asymmetry in attribution space. Asymmetries also appear in seasonal inter-basin interaction patterns.

**Rivera Tello et al. 2023** — LRP distinguishes strong EP El Niño from Central Pacific El Niño and La Niña using different precursor footprints; recovers ENSO diversity asymmetry in addition to phase asymmetry.

### Colfescu et al. 2024
**A Machine Learning-Based Approach to Quantify ENSO Sources of Predictability**
- **Journal:** Geophysical Research Letters, vol. 51, no. 13
- **DOI:** [10.1029/2023GL105194](https://doi.org/10.1029/2023GL105194)

Uses ML attribution to decompose ENSO predictability into contributions from SST, ocean heat content, and near-surface zonal winds (U10). Finds that U10 alone achieves skill comparable to SST at 11–21 month leads via an Indian Ocean atmospheric bridge — consistent with Bjerknes feedback operating through the Indo-Pacific coupled system. Shows ML attribution can recover the causal chain of coupled feedbacks even without explicit physics constraints.

### Deep Learning for Initial Error Sensitivity — 2025
**Using Deep Learning to Identify Initial Error Sensitivity for Interpretable ENSO Forecasts**
- **Journal:** Artificial Intelligence for the Earth Systems (AIES), vol. 4, no. 2
- **DOI:** [10.1175/AIES-D-24-0045.1](https://doi.org/10.1175/AIES-D-24-0045.1)
- **arXiv:** [2404.15419](https://arxiv.org/abs/2404.15419)

Uses an optimized model-analog method to estimate initial-error-sensitive regions. Finds El Niño forecasts are most sensitive to initial errors in tropical Pacific SST in boreal winter, while La Niña forecasts are more sensitive to zonal wind stress errors in boreal summer — directly analogous to what a SHAP conditional decomposition (El Niño vs. La Niña samples) would reveal.

---

## 5. CMIP6 Pretraining + Reanalysis Fine-Tuning

CMIP pretraining → reanalysis fine-tuning is the universal standard in the field:

| Paper | Pretraining data | Fine-tuning data | Notes |
|---|---|---|---|
| Ham et al. 2019 | 21 CMIP5 models | SODA 1871–1973 | Established paradigm |
| Lguensat et al. 2022 | Multiple CMIP6 | ORAS5 + ERA5 | Adds subsurface heat content |
| CTEFNet 2025 | 18 CMIP6 models | ORAS5 (1958–1978), ERA5 | Partial SPB mitigation |
| 3D-STransformer 2025 | 23 CMIP6 models | SODA 1871–1979 | 3D ocean state |

**CMIP6 data is available on GLADE at `/glade/collections/cmip6/`.**
Recommended models: CESM2, MPI-ESM1-2-LR, MIROC6.

### Hybrid Deep Learning in Low-Data Regime — 2024
- **arXiv:** [2412.03743](https://arxiv.org/abs/2412.03743)

Addresses the core problem of insufficient observational record (~45 years of ERA5) by combining physics-informed constraints with CMIP6 transfer learning. Directly justifies the CMIP6 → reanalysis design choice.

---

## 6. Bjerknes Feedback and Recharge Oscillator Recovery by XAI

### ENSO-PhyNet — 2024 (Most Physics-Constrained Model)
**Incorporating heat budget dynamics in a Transformer-based deep learning model for skillful ENSO prediction**
- **Journal:** npj Climate and Atmospheric Science
- **DOI:** [10.1038/s41612-024-00741-y](https://doi.org/10.1038/s41612-024-00741-y)

ENSO-PhyNet embeds Bjerknes feedback terms — zonal advection, thermocline feedback, Ekman pumping — as physics-informed constraints directly in transformer self-attention blocks. Attention weights are interpretable as physical process contributions. Achieves 22-month lead skill and explicitly shows that **thermocline feedback dominates attention at 12–18 month leads**. The most physics-constrained interpretable ENSO model to date.

---

### 3D-Geoformer — 2025
**The 3D-Geoformer for ENSO studies: a Transformer-based model with integrated gradient methods for enhanced explainability**
- **Journal:** Journal of Oceanology and Limnology, vol. 43, pp. 1688–1708
- **DOI:** [10.1007/s00343-025-4330-y](https://doi.org/10.1007/s00343-025-4330-y)

Applies Integrated Gradients to a 3D transformer ENSO model. IG maps show that **subsurface heat content in the western/central equatorial Pacific** — the recharge oscillator's discharge/recharge state variable — is the dominant contributor at long lead times (>12 months), while SST patterns dominate at short leads. The clearest demonstration of XAI recovering the recharge oscillator in a deep learning model.

---

### Mamalakis et al. 2022 — XAI Fidelity Benchmarks (Critical Methodological Reference)
**Investigating the Fidelity of Explainable Artificial Intelligence Methods for Applications of Convolutional Neural Networks in Geoscience**
- **Authors:** Mamalakis, A., Ebert-Uphoff, I. et al.
- **Journal:** Artificial Intelligence for the Earth Systems (AIES), vol. 1, no. 4
- **DOI:** [10.1175/AIES-D-22-0012.1](https://doi.org/10.1175/AIES-D-22-0012.1)
- **arXiv:** [2202.03407](https://arxiv.org/abs/2202.03407)

Benchmarks LRP, Integrated Gradients, SHAP, saliency maps, and GradCAM for fidelity in geoscience CNN applications. Identifies failure modes including gradient shattering, sign ambiguity, and zero-input blindness. **Must read before finalizing which SHAP explainer variant to use** — provides theoretical grounding for choosing TreeExplainer vs. GradientExplainer vs. KernelSHAP and for validating that SHAP recovers true model reasoning rather than attribution artifacts.

---

## Positioning for This Project

### Novelty
1. **SHAP as the primary XAI tool** — not IG or LRP; TreeExplainer provides exact (not approximate) Shapley values for XGBoost, enabling the most rigorous feature attribution available for the baseline model.
2. **Spring barrier via seasonal SHAP decomposition** — explicitly quantifying which features lose or retain importance through MAM initialization months, and computing a spring barrier index (MAM/DJF SHAP ratio per feature).
3. **Systematic multi-lead, multi-model comparison** — XGBoost → LSTM → CNN with consistent SHAP analysis at 3, 6, and 12-month leads, enabling a clean comparison of feature attribution across architectures.

### Physical Validation Strategy
SHAP maps should be validated against what ResoNet's IG and 3D-Geoformer's IG already found:
- **Short leads (3 months):** SST anomaly patterns dominate
- **Long leads (12 months):** Subsurface heat content / D20 (recharge oscillator state) dominates
- **Thermocline feedback** (Bjerknes chain) should appear in wind stress × D20 interaction SHAP values

Agreement with IG/LRP results validates SHAP fidelity; disagreements may constitute scientific findings.

---

## Open-Source Code Repositories

| Repository | Paper / Purpose | URL |
|---|---|---|
| Ham lab DL_ENSO | Ham et al. 2019 — foundational CMIP5→SODA CNN | https://github.com/jeonghwan723/DL_ENSO |
| Ham lab A_CNN | Ham et al. 2021 (Science Bulletin follow-up) | https://github.com/jeonghwan723/A_CNN |
| ENSO reproduction + GradCAM | Community reproduction of Ham 2019 | https://github.com/ZiluM/Deep-learning-for-multi-year-ENSO-Reproduction |
| ResoNet | Mu et al. 2024 | See arXiv:2312.10429 |
| OceanHackWeek ENSO | Community tutorial project 2022 | https://github.com/oceanhackweek/ohw22-proj-ENSO_Prediction |
