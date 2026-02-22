# State-of-the-Art: RuleFit, Rule Ensembles & Interpretable ML for Regulated Environments

> Last updated: February 2026. Covers research from 2023--2026 relevant to ExpertRuleFit.

---

## 1. RuleFit Extensions & Competing Rule-Based Methods

### LIRE: Learning Locally Interpretable Rule Ensemble (2023)
- **Authors:** Kentaro Kanamori (Fujitsu Limited)
- **Venue:** ECMLPKDD 2023; [arXiv:2306.11481](https://arxiv.org/abs/2306.11481)
- **Key Contribution:** Introduces "local interpretability" -- evaluated by the total number of rules needed to explain individual predictions rather than the model as a whole. Uses a novel regularizer promoting local interpretability, solved via coordinate descent with local search. The average number of weighted rules per prediction drops from 3.8 (RuleFit) to 1.1 (LIRE) while maintaining comparable accuracy.
- **Code:** [github.com/kelicht/lire](https://github.com/kelicht/lire)

### Causal Rule Ensemble for Heterogeneous Treatment Effects (2023--2024)
- **Authors:** Mayu Hiraishi, Ke Wan, Kensuke Tanioka, Hiroshi Yadohisa, Toshio Shimokawa
- **Venue:** *Statistical Methods in Medical Research*, Vol 33(6), pp. 1021--1042, 2024
- **Key Contribution:** Extends RuleFit for causal inference in randomized clinical trials. The ensemble comprises prognostic rules, prescriptive rules, and linear effects. By including a prognostic term, the selected rules represent heterogeneous treatment effects that exclude confounding main effects. A related earlier paper by Wan et al. (2023, *Statistics in Medicine*) introduced rule ensembles with adaptive group lasso for HTE estimation.
- **URL:** [journals.sagepub.com/doi/10.1177/09622802241247728](https://journals.sagepub.com/doi/10.1177/09622802241247728)

### Survival Causal Rule Ensemble (2024)
- **Authors:** Ke Wan, Kensuke Tanioka, Toshio Shimokawa
- **Venue:** *Statistics in Medicine*, Vol 43(27), pp. 5234--5271, 2024
- **Key Contribution:** Extends the causal rule ensemble framework to survival outcomes, enabling interpretable heterogeneous treatment effect estimation in time-to-event data.

### GWO+RuleFit: Heuristic-Optimized Rule Selection (2024)
- **Venue:** [PMC (2024)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11338282/)
- **Key Contribution:** Replaces Lasso-based rule selection in RuleFit with the Gray Wolf Optimizer (GWO), a metaheuristic search algorithm. Applied to mid-chemoradiation FDG-PET response prediction. Addresses RuleFit's tendency to generate too many rules, reducing overfitting and improving interpretability.

### Prediction Rule Ensembles with Relaxed/Adaptive Lasso (2023)
- **Authors:** Marjolein Fokkema, A. Hilbert
- **Venue:** CMStatistics 2023 (16th International Conference of the ERCIM WG)
- **Key Contribution:** Applies relaxed and adaptive lasso penalties to prediction rule ensembles (PREs) to improve sparsity and reduce the false-positive selection rate inherent to standard Lasso. The relaxed lasso allows retaining a pre-specified low number of terms with adequate accuracy.
- **Software:** R package `pre` (v1.0.8, CRAN); [github.com/marjoleinF/pre](https://github.com/marjoleinF/pre)

### Interpretable Prediction Rule Ensembles with Missing Data (2024)
- **Authors:** Fokkema, Schroeder, Schwerter, Doebler
- **Venue:** [arXiv:2410.16187](https://arxiv.org/abs/2410.16187) (October 2024)
- **Key Contribution:** First systematic study of PREs in the presence of missing data. Uses data stacking with multiple imputation instead of traditional model pooling, with relaxed Lasso to reduce bias.

### Spline-Rule Ensemble with Structured Sparsity (SRE-SGL, 2021)
- **Authors:** Koen W. De Bock, Arno De Caigny
- **Venue:** *Decision Support Systems*, Vol 150, 2021
- **Key Contribution:** Extends rule ensembles with spline basis functions and sparse group lasso (SGL) regularization. Groups rule, spline, and linear terms by the variables they depend on, enforcing structured sparsity both between and within term groups. Demonstrated superior AUC and interpretability on 14 customer churn datasets.
- **URL:** [sciencedirect.com/science/article/abs/pii/S0167923621000336](https://www.sciencedirect.com/science/article/abs/pii/S0167923621000336)

### SIRUS: Stable and Interpretable RUle Set (2021, extensions through 2024)
- **Authors:** Clement Benard, Gerard Biau, Sebastien da Veiga, Erwan Scornet
- **Venue:** *Electronic Journal of Statistics*, 15(1), pp. 427--505, 2021
- **Key Contribution:** Aggregates forest structure rather than predictions -- selecting the most frequent nodes to form a stable rule ensemble. Achieves dramatically better stability than RuleFit (running twice on the same dataset produces nearly identical rule lists), with predictive accuracy close to random forests. Includes an automatic stopping criterion based on stability (stops growing when 95% of rules are identical across runs). Extended to regression (AISTATS 2021) and spatial data (S-SIRUS, 2024).
- **S-SIRUS (2024):** Combines SIRUS with RF-GLS for spatial regression, outperforming SIRUS in simplicity and stability when spatial dependence exists. [arXiv:2408.05537](https://arxiv.org/html/2408.05537)
- **Software:** R package `sirus` (CRAN); Julia package `SIRUS.jl` (Huijzer et al., JOSS 2023)
- **URL:** [arXiv:1908.06852](https://arxiv.org/abs/1908.06852)

### FIGS: Fast Interpretable Greedy-Tree Sums (2025, PNAS)
- **Authors:** Yan Shuo Tan, Chandan Singh, Keyan Nasseri, Abhineet Agarwal, Bin Yu
- **Venue:** *Proceedings of the National Academy of Sciences* (PNAS), February 2025
- **Key Contribution:** Generalizes CART to simultaneously grow a flexible number of trees in summation. At each iteration, FIGS greedily selects whichever rule (across all trees) reduces unexplained variance most. The number and shape of trees emerge automatically from data. Achieves state-of-the-art performance when restricted to few splits (<20). Disentangles additive model components. Bagging-FIGS variant achieves competitive performance with random forests and XGBoost.
- **Software:** Part of `imodels` Python package
- **URL:** [pnas.org/doi/10.1073/pnas.2310151122](https://www.pnas.org/doi/10.1073/pnas.2310151122)

### Compressed Rule Ensemble (CRE, 2022)
- **Authors:** Malte Nalenz, Thomas Augustin (LMU Munich)
- **Venue:** AISTATS 2022, PMLR 151:9998--10014
- **Key Contribution:** Compresses clusters of similar rules into "ensemble rules" with soft (pooled) outputs, preserving the smoothing behavior lost when reducing to a small set of hard-threshold rules. CRE clearly outperforms RuleFit in sparsity and is competitive with SIRUS while preserving better AUC.
- **URL:** [proceedings.mlr.press/v151/nalenz22a.html](https://proceedings.mlr.press/v151/nalenz22a.html)

### FIRE: Fast Interpretable Rule Extraction (KDD 2023)
- **Authors:** Brian Liu, Rahul Mazumder
- **Venue:** ACM SIGKDD 2023
- **Key Contribution:** Optimization-based framework using fusion regularization (encouraging shared antecedents across rules) plus a non-convex sparsity-inducing penalty. Develops a specialized block coordinate descent solver that is up to 40x faster than existing solvers. Outperforms state-of-the-art rule ensemble algorithms at building sparse rule sets.
- **URL:** [arXiv:2306.07432](https://arxiv.org/abs/2306.07432)

### Unified Integer Programming Approach for Rule Extraction (2024--2025)
- **Authors:** Lorenzo Bonasera, Stefano Gualandi
- **Venue:** [arXiv:2407.00843](https://arxiv.org/abs/2407.00843) (July 2024); *Computers & Operations Research* (September 2025)
- **Key Contribution:** Formulates rule extraction from tree ensembles as a set partitioning problem via integer programming. Produces unweighted rule lists defining a partition of training data (each instance assigned to exactly one rule). Works with tabular and time series data. Combines ideas from FIRE (fusion penalty) and SIRUS (occurrence frequency). No parameter tuning needed under common settings.

### Optimal Rule Extraction via Discrete Optimization (2025)
- **Venue:** [arXiv:2506.20114](https://arxiv.org/html/2506.20114)
- **Key Contribution:** Proposes an estimator that jointly controls both the number of rules and their interaction depths. Develops a specialized exact algorithm solving the discrete optimization to global optimality, scaling beyond commercial solvers (Gurobi, Mosek). Provides regularization paths allowing practitioners to assess complexity vs. accuracy trade-offs.

---

## 2. EBM / GAM Advances

### EBM with Sparsity (2023)
- **Authors:** Greenwell et al.
- **Key Contribution:** LASSO pruning reduces EBM terms by 80-95% without accuracy loss.

### Cross Feature Selection for EBM (2023)
- **Authors:** Charran, Mahapatra
- **Key Contribution:** Fixes spurious interactions and single-feature dominance in EBM interaction terms.

### LLMs Understand Glass-Box Models (2023)
- **Authors:** Lengerich, Nori, Caruana
- **Key Contribution:** LLMs can reason about EBM shape functions, detect anomalies, and suggest repairs.

### TalkToEBM (AAAI-24 Workshop, 2024)
- **Authors:** Bordt et al.
- **Key Contribution:** Open-source LLM-GAM interface for natural language model interaction with EBMs.

### IGANN: Interpretable Glassbox Based on Augmented Neural Networks (EJOR 2024)
- **Authors:** Kraus et al.
- **Key Contribution:** Neural GAM with smoother shape functions than EBM's step functions.

### GP-NAM: Gaussian Process Neural Additive Models (AAAI 2024)
- **Authors:** Zhang et al.
- **Key Contribution:** Gaussian Process GAM via convex optimization -- no randomness in shape functions.

### Transparent ML with EBM in Earth Observation (2025)
- **Key Contribution:** Human-machine collaboration where domain experts directly edit EBM shape functions for satellite imagery classification.

### Explainable Boosting Machines (EBMs / GA2M)
- **Authors:** Microsoft Research (Nori et al., Lou et al.)
- **Status:** Actively developed ([InterpretML](https://interpret.ml/docs/ebm.html)); R package on CRAN as of July 2025
- **Key Contribution:** Tree-based cyclic gradient boosting GAM with automatic pairwise interaction detection. Often as accurate as black-box models while remaining fully interpretable (glass-box). Recent work (2024--2025) addresses spurious interactions and single-feature dominance in interaction terms. Fast at prediction time (lookups + additions).

---

## 3. Stability & Reproducibility in Feature Selection

### LASSO and Elastic Net Over-Select (2023)
- **Authors:** Liu et al.
- **Venue:** MDPI 2023
- **Key Finding:** Both LASSO and Elastic Net select too many features; Elastic Net selects *even more*.

### Weighted Stability Selection (2023)
- **Venue:** [*Scientific Reports*](https://www.nature.com/articles/s41598-023-32517-4), 2023
- **Key Contribution:** Extends stability selection by weighting variables using AUC from additional modeling. Achieves higher AUC with fewer selected variables compared to standard stability selection.

### Integrated Path Stability Selection (IPSS, 2023)
- **Key Contribution:** Orders of magnitude tighter bounds on false positives than the original stability selection of Meinshausen & Buhlmann (2010).

### Automated Calibration for Stability Selection (`sharp` R package, 2023)
- **Key Contribution:** Automated threshold calibration for stability selection.

### VSOLassoBag (2023)
- **Authors:** Liang et al.
- **Key Contribution:** Bagging + LASSO: bootstrap frequency for stable variable selection in omics. Uses essentially the same paradigm as ExpertRuleFit's bootstrap frequency filtering.

### On the Selection Stability of Stability Selection (2024)
- **Venue:** [arXiv:2411.09097](https://arxiv.org/html/2411.09097v1), November 2024
- **Key Contribution:** Revisits the asymptotic PFER upper-bound from Meinshausen & Buhlmann. Derives a point-wise control version. Establishes a two-way control: fixing the threshold determines the upper-bound, and vice versa. Extends the Shah & Samworth (2013) complementary pairs bootstrap bounds.

### Bootstrapping Lasso in Generalized Linear Models (2024)
- **Venue:** [arXiv:2403.19515](https://arxiv.org/abs/2403.19515)
- **Key Contribution:** Develops Perturbation Bootstrap and Pearson's Residual Bootstrap methods for approximating the distribution of the Lasso estimator in GLMs, enabling valid statistical inference for sub-models. Directly relevant to the Lasso step in RuleFit.

### Stochastic LASSO for High-Dimensional Data (2025--2026)
- **Venue:** [*Scientific Reports*](https://www.nature.com/articles/s41598-026-35273-3), 2026
- **Key Contribution:** Proposes Stochastic LASSO to address multicollinearity within bootstrap samples, missing predictors in draws, and randomness in predictor sampling.

### Stabilizing ML for Reproducible and Explainable Results (2025)
- **Venue:** [*Computer Methods and Programs in Biomedicine*](https://www.sciencedirect.com/science/article/pii/S0169260725003165), 2025
- **Key Contribution:** Novel validation approach to stabilize predictive performance and feature importance at both group and subject-specific levels when ML models are initialized with stochastic processes and random seeds.

---

## 4. Rashomon Sets & Model Multiplicity

### Rashomon Importance Distribution (RID) (NeurIPS 2023 Spotlight)
- **Authors:** Donnelly, Rudin
- **Key Contribution:** Variable importance across the full Rashomon set, not just one model.

### "Amazing Things Come From Having Many Good Models" (ICML 2024)
- **Authors:** Rudin et al.
- **Key Contribution:** Position paper: the Rashomon set as a space of possibilities for fairness, monotonicity, and other constraints.

### "The Rashomon Set Has It All" (2025)
- **Venue:** OpenReview 2025
- **Key Contribution:** Enumerating near-optimal sparse decision trees satisfies 7 trustworthiness metrics simultaneously.

### Double-Edged Nature of the Rashomon Set (2025)
- **Key Contribution:** Conflicting feature attributions across equally-good models undermine trust. Highlights the need for methods that collapse model multiplicity.

### Rashomon Sets in Federated Learning (2026)
- **Key Contribution:** First formalization in federated learning: global vs client-specific Rashomon sets.

**Relevance to ExpertRuleFit:** Standard RuleFit's 1/100 stability means it samples different models from the Rashomon set each run. ExpertRuleFit collapses this multiplicity to a single deterministic point -- a strong theoretical framing for reproducibility as Rashomon set selection.

---

## 5. Regulatory AI & Credit Scoring

| Publication | Source | Year | Key Takeaway |
|-------------|--------|------|-------------|
| EU AI Act: Credit scoring = high-risk AI | Regulation EU 2024/1689 | 2024 | Full applicability Aug 2026; transparency + human oversight mandated |
| EBA Follow-Up on ML for IRB Models | EBA/REP/2023/28 | 2023 | 40% of institutions use Shapley values; lack of clarity on supervisory expectations |
| ECB Guide to Internal Models -- ML Section | ECB | July 2025 | First explicit ECB expectations for ML: justification of complexity, explainability tools, audit plans |
| CSSF Second Thematic Review on AI | CSSF Luxembourg | May 2025 | 28% of institutions have AI in production; governance + explainability emphasized |
| BIS: Managing Explanations | FSI Paper No. 24 | 2025 | Recommends allowing complex models if adequate safeguards exist |
| Deterministic Reproducibility in Financial AI | IJRAI | 2026 | Reproducibility as an architectural property, not just model property |
| ECB Banking Supervision Newsletter | ECB | Nov 2025 | 54% of European banks use AI for credit scoring; decision trees most common |

---

## 6. Weighted Regularization for Prior Knowledge

### LLM-Lasso (Stanford, 2025)
- **Key Contribution:** LLM-derived penalty factors in Lasso; features identified as important get lower penalties. Opens a future direction: using LLMs to automatically set penalty weights for ExpertRuleFit.

### Weighted Lasso for Known Regressors (INRIA, 2024)
- **Key Contribution:** Penalty=0 for known-important features; matches oracle performance at >90% prior knowledge. Provides direct theoretical backing for ExpertRuleFit's confirmatory rule mechanism (w_j = 1e-8).

### Prior Knowledge in Regularized Regression (Zeng et al., 2021)
- **Key Contribution:** Feature-specific penalties as log-linear functions of meta-features via empirical Bayes.

### Monotonic Neural Additive Models (Yang et al., 2022)
- **Key Contribution:** Monotonicity constraints for credit scoring; matches black-box accuracy.

### Equifax Regulatory Sandbox (2024)
- **Key Contribution:** ML with monotonic constraints beats logistic regression, matches explainability requirements.

---

## Summary: Competitive Landscape

| Theme | Most Impactful Methods | Key Advantage Over RuleFit |
|-------|----------------------|---------------------------|
| **Stability** | SIRUS, Bayes Point Rule Classifier | Dramatically more stable rule sets across runs |
| **Sparsity** | CRE, FIRE, LIRE | Fewer/simpler rules with comparable accuracy |
| **Causal Inference** | Causal Rule Ensemble (Hiraishi et al.) | Separates prognostic from prescriptive effects |
| **Optimal Rule Extraction** | IP-based methods (Bonasera), FIRE | Global optimality guarantees for rule selection |
| **Additive Structure** | FIGS, EBMs | Explicit additive decomposition with interpretability |
| **Bootstrap Stabilization** | Stability Selection, Weighted SS, Stochastic LASSO | Stable feature selection under Lasso regularization |
| **Missing Data** | PRE with MI (Fokkema et al. 2024) | Robust rule ensembles with incomplete data |

---

## ExpertRuleFit Positioning

**Unique niche:** No existing method combines (a) RuleFit rule generation + (b) bootstrap stability selection + (c) weighted regularization for confirmatory rules + (d) EBM interaction discovery. Each component exists in isolation in the literature, but the integration is novel.

**Theoretical framing:** The reproducibility problem can be framed as Rashomon set collapse (Rudin et al. 2024) -- ExpertRuleFit deterministically selects one point from the Rashomon set.

**Closest competitors:**
- **SIRUS** -- stability via quantile discretization (but no expert rule preservation)
- **FIRE / CRE** -- sparsity (but no bootstrap stabilization or regulatory compliance)
- **Stability Selection** -- bootstrap filtering (but applied to generic Lasso, not rule ensembles with confirmatory constraints)

**Regulatory timing:** The ECB July 2025 Guide + EU AI Act Aug 2026 applicability create a direct window for ExpertRuleFit's value proposition.

**Future directions from SOTA:**
- LLM integration (TalkToEBM) for automated rule explanation
- Causal rule extensions for treatment effect estimation
- GP-NAM determinism as an alternative to EBM
- IPSS tighter stability bounds for bootstrap filtering
- Integer programming for globally optimal rule subset selection

---

## Key References

1. Friedman & Popescu (2008) -- *Predictive Learning via Rule Ensembles*, Annals of Applied Statistics
2. Meinshausen & Buhlmann (2010) -- *Stability Selection*, JRSS-B, 72(4), pp. 417--473
3. Benard et al. (2021) -- *SIRUS: Stable and Interpretable RUle Sets*, EJS, 15(1), pp. 427--505
4. Nalenz & Augustin (2022) -- *Compressed Rule Ensembles*, AISTATS (PMLR 151)
5. Kanamori (2023) -- *LIRE: Learning Locally Interpretable Rule Ensemble*, ECMLPKDD
6. Liu & Mazumder (2023) -- *FIRE: Fast Interpretable Rule Extraction*, KDD
7. Hiraishi et al. (2024) -- *Causal Rule Ensemble for HTE*, Stat Methods Med Res
8. Fokkema et al. (2024) -- *Interpretable PREs with Missing Data*, arXiv:2410.16187
9. Tan et al. (2025) -- *FIGS*, PNAS
10. Rudin et al. (2024) -- *Amazing Things Come From Having Many Good Models*, ICML
11. Nori et al. (2019) -- *InterpretML*, arXiv:1909.09223
12. Singh et al. (2021) -- *imodels*, JOSS
