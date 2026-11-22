# Introduction

RuleFit turns the nodes of a tree ensemble into binary rules and selects
a sparse subset with an $\ell_1$ penalty. The fitted model is an
intercept plus a weighted sum of a few rules and linear terms, so each
prediction can be traced to the conditions that produced it.

The selection step is unstable. Candidate rules are strongly correlated,
and among correlated predictors the Lasso keeps one more or less
arbitrarily. A different seed for the tree ensemble or for the
cross-validation folds therefore gives a different rule set on the same
data. In our benchmark
(Section <a href="#sec:exp" data-reference-type="ref"
data-reference="sec:exp">5</a>), standard RuleFit returned the same rule
set in 1 run out of 100, with a mean pairwise Jaccard similarity of
about 0.05.

In regulated settings such as credit, insurance or tax administration,
this blocks deployment for three reasons:

- **Re-execution.** A supervisor who re-runs the model must obtain the
  same output, otherwise decisions based on it cannot be audited.

- **Explanation.** A rule given to an analyst as the reason for a score
  should not disappear at retraining for reasons unrelated to the data.

- **Expert knowledge.** Criteria required by regulation or by business
  experts can be removed by the penalty when their signal is weak or
  collinear with other rules.

#### Contributions.

1.  An estimator combining RuleFit rule generation, bootstrap
    selection-frequency filtering and penalty-weighted expert rules,
    with a constrained refit that guarantees the inclusion of
    confirmatory rules
    (Section <a href="#sec:method" data-reference-type="ref"
    data-reference="sec:method">3</a>).

2.  A precise statement of the reproducibility guarantee as determinism
    within an execution envelope, and of what it does not cover
    (Section <a href="#sec:repro" data-reference-type="ref"
    data-reference="sec:repro">4</a>).

3.  A benchmark on three credit datasets, together with an explicit
    analysis of what it does and does not show, and a protocol for
    measuring stability under data perturbation
    (Section <a href="#sec:exp" data-reference-type="ref"
    data-reference="sec:exp">5</a>).

4.  An open-source Python implementation (about 4,000 lines, 39
    automated tests) built on `imodels` and `scikit-learn`.

# Related work

#### Stable rule ensembles.

SIRUS obtains stability from the forest structure: splits are
discretised on quantiles and the most frequent rules across trees are
kept. Two runs on the same data yield nearly the same rule list, and
stability under data perturbation is reported. Other methods target
sparsity: Compressed Rule Ensembles merge similar rules, FIRE uses a
fusion penalty favouring shared antecedents, LIRE minimises the number
of rules needed per prediction, and FIGS grows a small sum of trees. The
`pre` package uses relaxed and adaptive Lasso to reduce false
selections.

#### Stability selection.

Retaining variables selected in most subsample fits is the stability
selection of , with improved error bounds in .
<span class="smallcaps">ExpertRuleFit</span> applies the idea to rule
ensembles with a fixed threshold and ten draws, without the error
control these works provide.

#### Prior knowledge in penalties.

Reducing the penalty on chosen variables follows the adaptive Lasso and
the use of per-feature penalty factors. Monotonicity constraints, common
in credit scoring, fix the direction of an effect;
<span class="smallcaps">ExpertRuleFit</span> enforces inclusion, not
direction.

#### Glass-box additive models.

Explainable Boosting Machines fit additive shape functions with pairwise
interactions; we use them to propose interaction rules and in a stacked
variant.

#### Model multiplicity.

Many models fit the data almost equally well . In that view, RuleFit’s
instability is a random draw from the Rashomon set, and
<span class="smallcaps">ExpertRuleFit</span> fixes the draw. This makes
the output auditable but does not show that the chosen model is better
than its neighbours.

# Method

Let $X\in\mathbb{R}^{n\times p}$ and $y\in\{0,1\}^n$.
<span class="smallcaps">ExpertRuleFit</span> fits a logistic model on an
augmented matrix
$$Z = [\,x_1,\dots,x_p,\; r_1(X),\dots,r_k(X),\; c_1(X),\dots,c_m(X),\; o_1(X),\dots,o_q(X)\,],$$
where $r_j$ are tree-derived binary rules, $c_j$ confirmatory expert
rules and $o_j$ optional expert rules. Predictions are
$P(y=1\mid x)=\sigma(\beta_0+\sum_j \beta_j z_j)$.

#### Penalty weighting.

Each column receives a weight $w_j$ (1 for linear terms and tree rules,
$w_c=10^{-8}$ for confirmatory rules, $w_o=0.3$ for optional rules by
default). Columns are rescaled as $\tilde z_j = z_j/\sqrt{w_j}$ before
an elastic-net fit, which scales the effective penalties:
$$\text{L1 term}\propto \sqrt{w_j}\,|\beta_j|,\qquad \text{L2 term}\propto w_j\,\beta_j^2 .$$
With $w_c=10^{-8}$ the $\ell_1$ penalty on confirmatory rules is reduced
by a factor of $10^4$. Coefficients are mapped back to the original
scale before reporting.

<div class="algorithm">

Fit gradient boosting (250 trees, depth 4, seed 42) and extract at most
50 rules $R$ via `imodels` RuleFit Build $Z$ from linear terms, $R$,
$C$, $O$; set weights $w$ Keep columns selected in at least $\tau B$
fits, plus all of $C$ Fit final elastic-net `LogisticRegressionCV` on
kept columns

</div>

The user’s `random_state` is ignored (a warning is emitted); all
internal seeds derive from a fixed base seed. Two extensions are
provided: `discover_interaction_rules`, which uses an EBM to find
interacting feature pairs and thresholds and returns them as *optional*
rules, and `DualModel`, which stacks EBM and
<span class="smallcaps">ExpertRuleFit</span> scores with a logistic
meta-model trained on out-of-fold predictions. Optional post-hoc
calibration (Platt or isotonic, 5-fold) is available but off by default.

#### Expert-imposed versus data-discovered rules.

Confirmatory status is reserved for rules an expert imposes. Rules found
by the EBM extension are data-discovered: they enter as optional rules,
with a reduced penalty, and must pass the frequency filter like any tree
rule. An expert can promote one to confirmatory after review; each
discovered rule records its provenance (`source="ebm_discovered"`) for
the audit trail.

#### Sign check and absorption diagnostics.

A confirmatory rule may declare an expected sign $s_c\in\{+1,-1\}$.
After fitting, the estimator compares $\operatorname{sign}(\beta_c)$
with $s_c$, sets a flag `confirmatory_sign_ok_` and emits a warning for
each contradiction; the sign is checked, not enforced. For each
confirmatory rule it also reports, on the training data, the largest
absolute correlation with another retained column, its variance
inflation factor, the retained rules with $|r|\ge 0.5$ that the final
model set to zero, and the change in their coefficients when the
confirmatory rule is removed (ridge refits with the final model’s $C$
and the same weights). These diagnostics make absorption visible; they
do not prevent it.

# Reproducibility guarantee

The guarantee is bitwise determinism within an execution envelope: the
same training data, in the same column order, on the same software
stack, yields the same rule set and predictions. The envelope requires
pinned versions of Python, NumPy, scikit-learn and imodels;
single-threaded BLAS and OpenMP; the fixed internal seeds; a validated
input schema with stable feature order; and identical platform settings.
A container recipe with pinned image digests is provided in the
repository.

The guarantee does not cover different data (adding or correcting rows
may change the rule set), a different software stack or CPU, or
dependence on row order where underlying algorithms break ties by
position. Determinism makes re-execution auditable; robustness to small
data changes, which governs whether explanations survive retraining, is
a distinct property.

# Experiments

#### Data and protocol.

We use German Credit (UCI, 1,000 rows), Taiwan Credit (UCI, 30,000 rows)
and HMEQ (about 5,960 rows). Each dataset is split once into 80%
training and 20% test data (stratified, seed 42). Each model is then
fitted 100 times with `random_state` $=0,\dots,99$ on that split. A
run’s rule set is the set of rules with non-zero coefficient.
*Stability* is the number of runs sharing the most frequent rule set;
Jaccard similarity is averaged over sampled pairs of runs.

<div id="tab:results">

|                    | Identical rule sets (of 100) |                                              | Test AUC |                                              |
|:-------------------|:----------------------------:|:--------------------------------------------:|:--------:|:--------------------------------------------:|
| 2-3(lr)4-5 Dataset |           RuleFit            | <span class="smallcaps">ExpertRuleFit</span> | RuleFit  | <span class="smallcaps">ExpertRuleFit</span> |
| German Credit      |              1               |                     100                      |  0.788   |                    0.801                     |
| Taiwan Credit      |              1               |                     100                      |  0.764   |                    0.763                     |
| HMEQ               |              1               |                     100                      |  0.903   |                    0.903                     |

Seed sensitivity and discrimination (single train/test split, 100
seeds).

</div>

Mean pairwise Jaccard similarity is about 0.05 for RuleFit and 1.000 for
<span class="smallcaps">ExpertRuleFit</span>; the standard deviation of
<span class="smallcaps">ExpertRuleFit</span>’s AUC across runs is 0.

#### What the results show.

RuleFit’s output depends heavily on the seed, so a single run cannot be
reproduced without recording every seed.
<span class="smallcaps">ExpertRuleFit</span> removes this dependence
without loss of discrimination on these datasets.

#### What they do not show.

Because <span class="smallcaps">ExpertRuleFit</span> ignores
`random_state`, varying it does not change the computation: the 100/100
result follows from the design and is not an empirical finding about
stability. The comparison sets determinism against seed sensitivity. AUC
is also reported on a single split without confidence intervals.

#### Proposed perturbation protocol.

To measure stability under data perturbation we will: (1) draw 100
bootstrap or 90% subsamples of the training set; (2) fit RuleFit with a
fixed seed and <span class="smallcaps">ExpertRuleFit</span> on each; (3)
report mean pairwise Jaccard similarity and per-rule selection
frequencies, as in ; (4) report AUC with confidence intervals over
repeated splits; (5) include SIRUS and $\ell_2$-penalised logistic
regression as baselines. These results are not yet available.

# Limitations

- **Stability under perturbation is unmeasured**
  (Section <a href="#sec:exp" data-reference-type="ref"
  data-reference="sec:exp">5</a>).

- **No sign constraint.** A confirmatory rule is guaranteed to stay in
  the model, not to keep the direction the expert intended. A declared
  expected sign is checked and a contradiction is flagged, but the
  coefficient is not constrained.

- **Unpenalised rules can absorb signal.** With $w_c=10^{-8}$, a
  confirmatory rule correlated with tree rules can take their weight; a
  weak or wrong expert rule then shapes the model. The diagnostics
  report this case; they do not correct it.

- **Calibration not evaluated.** Penalised logistic outputs are often
  too conservative; optional post-hoc calibration exists but reliability
  curves and Brier scores have not been reported.

- **Coarse frequency estimate.** With $B=10$, the 80% threshold means 8
  of 10 draws; the draws are fixed, not resampled.

- **Scope.** Binary classification only; evidence limited to three
  credit datasets of moderate size; rule readability (up to four
  conditions) not tested with users.

- **Review.** Single author, internal code review only; the
  implementation was written with AI assistance.

# Application design: tax-audit prioritisation

A tax administration ranking returns for review needs a reason an agent
can state, a selection that can be replayed if contested, and
inspectors’ criteria kept in the model. We sketch a design; the method
has not been applied to tax data.

| Model element         | Tax-audit counterpart                                                                                                                |
|:----------------------|:-------------------------------------------------------------------------------------------------------------------------------------|
| Target $y$            | Audit outcome, e.g. adjustment above a threshold set with the business                                                               |
| Tree-derived rules    | Patterns in past audit results, e.g. high deductions combined with mainly salaried income                                            |
| Confirmatory rules    | Legal or business criteria, e.g. a gap between declared salary and the employer’s statement, or deductions above a statutory ceiling |
| Optional rules        | Inspectors’ hypotheses, kept only if supported by data                                                                               |
| Determinism           | The selection list can be replayed for a contested case                                                                              |
| Active rules per case | The explanation given to the reviewing agent                                                                                         |

Mapping of model elements to a tax-audit setting (illustrative).

Three conditions precede any pilot. First, *selection bias*: outcomes
exist only for audited files, so a naive model learns past selection
habits; a small random audit sample and inverse-probability weighting
are needed. Second, *fairness and legal grounds*: variables without
legal basis are excluded and score distributions compared across
taxpayer groups. Third, *human decision*: the score suggests a file and
an agent decides, consistent with Article 22 GDPR on automated
decisions. A pilot would compare the hit rate per audit hour with the
current selection method on the random sample.

# Conclusion and future work

<span class="smallcaps">ExpertRuleFit</span> makes rule-ensemble outputs
exactly reproducible and keeps expert-imposed rules, at no loss of AUC
on three credit datasets. Its central empirical claim so far is
determinism, not stability under data change. Future work, in order of
priority: the perturbation protocol of
Section <a href="#sec:exp" data-reference-type="ref"
data-reference="sec:exp">5</a> with SIRUS as baseline; sign-constrained
confirmatory rules via bounded optimisation; a principled choice of the
confirmatory weight; more bootstrap draws or error-controlled thresholds
; calibration assessment; and handling of selection bias when labels
come only from past audits.

#### Code and data availability.

Code, tests, the validation script and a container recipe are at
<https://github.com/hdjebar/ExpertRuleFit>. Datasets are public.

<div class="thebibliography">

99 Bénard, C., Biau, G., da Veiga, S. and Scornet, E. (2021). SIRUS:
Stable and interpretable RUle set for classification. *Electronic
Journal of Statistics*, 15(1), 427–505. Fokkema, M. (2020). Fitting
prediction rule ensembles with R package pre. *Journal of Statistical
Software*, 92(12). Friedman, J. H. and Popescu, B. E. (2008). Predictive
learning via rule ensembles. *Annals of Applied Statistics*, 2(3),
916–954. Kanamori, K. (2023). Learning locally interpretable rule
ensemble. In *ECML PKDD*. Liu, B. and Mazumder, R. (2023). FIRE: An
optimization approach for fast interpretable rule extraction. In *KDD*.
Lou, Y., Caruana, R., Gehrke, J. and Hooker, G. (2013). Accurate
intelligible models with pairwise interactions. In *KDD*. Meinshausen,
N. and Bühlmann, P. (2010). Stability selection. *Journal of the Royal
Statistical Society B*, 72(4), 417–473. Nalenz, M. and Augustin, T.
(2022). Compressed rule ensemble. In *AISTATS*, PMLR 151. Nori, H.,
Jenkins, S., Koch, P. and Caruana, R. (2019). InterpretML: A unified
framework for machine learning interpretability. arXiv:1909.09223.
Rudin, C. et al. (2024). Amazing things come from having many good
models. In *ICML*. Shah, R. D. and Samworth, R. J. (2013). Variable
selection with error control: another look at stability selection.
*Journal of the Royal Statistical Society B*, 75(1), 55–80. Singh, C.,
Nasseri, K., Tan, Y. S., Tang, T. and Yu, B. (2021). imodels: a Python
package for fitting interpretable models. *Journal of Open Source
Software*, 6(61). Tan, Y. S., Singh, C., Nasseri, K., Agarwal, A. and
Yu, B. (2025). Fast interpretable greedy-tree sums. *PNAS*. Zou, H.
(2006). The adaptive lasso and its oracle properties. *Journal of the
American Statistical Association*, 101(476), 1418–1429.

</div>
