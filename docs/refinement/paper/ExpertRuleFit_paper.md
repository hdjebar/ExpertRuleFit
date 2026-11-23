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
data-reference="sec:exp">5</a>), standard RuleFit returned a different
rule set for each of 100 seeds, with a mean pairwise Jaccard similarity
of 0.04–0.06.

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
    with a constrained refit that enforces the inclusion of confirmatory
    rules in the sense of
    Definition <a href="#def:inclusion" data-reference-type="ref"
    data-reference="def:inclusion">1</a>: present with a non-zero
    coefficient, not with a given sign
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

4.  An open-source Python implementation built on `imodels` and
    `scikit-learn`. All results refer to
    <span class="smallcaps">ExpertRuleFit</span> 0.7.0, commit
    `281d5a6`, with 66 automated tests passing in a container built from
    `python:3.11-slim@sha256:07d48240…` and a hash-locked dependency
    file (SHA-256 `990ced9d…`).

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

#### Standardisation and penalty weighting.

Columns are first standardised with statistics from the training data:
linear terms are centred and divided by their standard deviation $s_j$,
rule columns are divided by $s_j$ only, which keeps them sparse. Each
column then receives a weight $w_j$ (1 for linear terms and tree rules,
$w_c=10^{-2}$ for confirmatory rules, $w_o=0.3$ for optional rules by
default) and is rescaled as $\tilde z_j = (z_j-m_j)/(s_j\sqrt{w_j})$
before an elastic-net fit, which scales the effective penalties on the
standardised coefficients:
$$\text{L1 term}\propto \sqrt{w_j}\,|\beta_j|,\qquad \text{L2 term}\propto w_j\,\beta_j^2 .$$
With $w_c=10^{-2}$ the $\ell_1$ penalty on confirmatory rules is reduced
by a factor of 10 and the $\ell_2$ penalty by a factor of 100.
Coefficients and the intercept are mapped back to the original scale
before reporting. Earlier versions used $w_c=10^{-8}$ without
standardisation. A confirmatory column was then $10^4$ times larger than
the others, which made the saga solver stop far from the optimum without
a convergence warning. After standardisation, the conditioning of the
problem depends on the ratio $1/\sqrt{w_c}$ rather than on the data;
$w_c=10^{-2}$ is the smallest value that fitted without convergence
warnings in the sensitivity study below. Inclusion of confirmatory rules
(Definition <a href="#def:inclusion" data-reference-type="ref"
data-reference="def:inclusion">1</a>) does not rely on $w_c$.

<div id="def:inclusion" class="definition">

**Definition 1** (Inclusion). A confirmatory rule $c$ is *included* if
its column is part of the final model and its fitted coefficient is
non-zero ($|\beta_c|>10^{-10}$).

</div>

Inclusion says nothing about the sign or size of $\beta_c$. The
estimator ensures inclusion by exempting $C$ from the frequency filter
and, if the final fit sets some $\beta_c$ to zero, refitting with $C$
unpenalised; if a rule is still excluded (perfect collinearity or a
constant column), the flag `confirmatory_all_active_` is false and a
warning is raised.

<div class="algorithm">

Fit gradient boosting (250 trees, depth 4, seed 42) and extract at most
50 rules $R$ via `imodels` RuleFit Build $Z$ from linear terms, $R$,
$C$, $O$; compute centres $m$, scales $s$ and weights $w$ Keep columns
selected in at least $\tau B$ fits, plus all of $C$ Fit final
elastic-net `LogisticRegressionCV` on the kept columns of
$(Z-m)/(s\sqrt{w})$

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
absolute correlation with any other candidate column, its variance
inflation factor among retained columns, the candidate columns with
$|r|\ge 0.5$ that are inactive in the final model (removed by the
frequency filter or set to zero), and the change in the coefficients of
retained correlated columns when the confirmatory rule is removed (ridge
refits with the final model’s $C$ and the same weights). These
diagnostics make absorption visible; they do not prevent it.

# Reproducibility guarantee

The guarantee is bitwise determinism within an execution envelope: the
same training data, in the same column order, on the same software
stack, yields the same rule set and predictions. The envelope requires
pinned versions of Python, NumPy, scikit-learn and imodels;
single-threaded BLAS and OpenMP; the fixed internal seeds; a validated
input schema with stable feature order; and identical platform settings.
A container recipe (base image pinned by digest, dependencies installed
from a hash-locked file) is provided in the repository.

The guarantee does not cover different data (adding or correcting rows
may change the rule set), a different software stack or CPU, or
dependence on row order where underlying algorithms break ties by
position. Identity across machines has not been tested: all results in
this report come from one x86-64 machine (Intel Xeon, 2.1 GHz), and runs
on another CPU type, even with the same container, may differ in the
last bits of floating-point results. Determinism makes re-execution
auditable; robustness to small data changes, which governs whether
explanations survive retraining, is a distinct property.

# Experiments

#### Data and protocol.

We use German Credit (UCI, 1,000 rows), Taiwan Credit (UCI; a random
subsample of 5,000 of the 30,000 rows, seed 42, to bound runtime) and
HMEQ (5,960 rows). Each dataset is split once into 80% training and 20%
test data (stratified, seed 42). Missing values are imputed (median, or
mode for categorical columns), categorical columns are ordinally encoded
and all columns are standardised. All preprocessing statistics are
fitted on the training split. RuleFit (imodels, 250 trees of depth 4, at
most 50 rules) is fitted 100 times with `random_state` $=0,\dots,99$.
<span class="smallcaps">ExpertRuleFit</span> is fitted with its default
settings and no expert rules, except for the solver tolerance: the
benchmark uses $10^{-6}$ instead of the library default $10^{-4}$, a
setting inherited from the original benchmark script and kept so that
the rerun changes only the preprocessing and the software stack. It is
fitted three times with `random_state` $=0,1,2$, each run in a separate
process; since the seed is ignored by design, more runs would repeat the
same computation, and we instead check that the three runs agree bit for
bit, comparing a hash of the rule set and the SHA-256 digest of the
test-set probabilities. A run’s rule set is the set of terms with
non-zero coefficient; we report how many RuleFit runs share the most
frequent rule set and the Jaccard similarity averaged over all pairs of
runs. All runs use the pinned software stack (Python 3.11.18,
scikit-learn 1.7.2, NumPy 2.4.6, imodels 3.0.4, single-threaded), in a
container built from the recipe in the repository. An earlier version of
this table fitted the preprocessing on the full dataset before the split
and used an unpinned stack.

<div id="tab:results">

|               |      Rule sets      |               |                                              |          Test AUC          |                                              |
|:--------------|:-------------------:|:-------------:|:--------------------------------------------:|:--------------------------:|:--------------------------------------------:|
| 2-4(lr)5-6    | RuleFit (100 seeds) |               | <span class="smallcaps">ExpertRuleFit</span> |                            |                                              |
| Dataset       |      distinct       | most frequent |                identical runs                |          RuleFit           | <span class="smallcaps">ExpertRuleFit</span> |
| German Credit |         100         |     1/100     |                     3/3                      | 0.788 ($\pm$<!-- -->0.009) |                    0.790                     |
| Taiwan Credit |         100         |     1/100     |                     3/3                      | 0.764 ($\pm$<!-- -->0.003) |                    0.769                     |
| HMEQ          |         100         |     1/100     |                     3/3                      | 0.903 ($\pm$<!-- -->0.003) |                    0.903                     |

Seed sensitivity and discrimination (single train/test split, no expert
rules). RuleFit: 100 seeds; AUC is the mean over seeds.
<span class="smallcaps">ExpertRuleFit</span>: three runs with
bitwise-identical output.

</div>

Mean pairwise Jaccard similarity of RuleFit rule sets is 0.045 (German
Credit), 0.058 (Taiwan Credit), 0.036 (HMEQ). The three
<span class="smallcaps">ExpertRuleFit</span> runs per dataset produced
the same rule set (43, 42, 48 terms) and the same probabilities. One
<span class="smallcaps">ExpertRuleFit</span> fit took between about 10
minutes (German Credit) and 3.5 hours (HMEQ) single-threaded, against
seconds for RuleFit.

#### What the results show.

Table <a href="#tab:results" data-reference-type="ref"
data-reference="tab:results">1</a> uses no expert rules: it compares
rule generation and selection only. RuleFit’s output depends heavily on
the seed, so a single run cannot be reproduced without recording every
seed. <span class="smallcaps">ExpertRuleFit</span> removes this
dependence without loss of discrimination on these datasets.

#### What they do not show.

Because <span class="smallcaps">ExpertRuleFit</span> ignores
`random_state`, varying it does not change the computation: identical
output across runs follows from the design and is not an empirical
finding about stability; the three runs only confirm that the
implementation is deterministic within the execution envelope. The
comparison sets determinism against seed sensitivity. AUC is also
reported on a single split without confidence intervals.

#### Penalty-weight sensitivity (German Credit).

We fitted <span class="smallcaps">ExpertRuleFit</span> on the same split
for $w_c\in\{10^{-8},10^{-6},10^{-4},10^{-2},1\}$ and
$w_o\in\{0.1,0.3,1\}$ on the pinned software stack, with the library’s
default solver tolerance ($10^{-4}$). The expert rules are standard
credit-risk criteria chosen by the author, not rules supplied by domain
experts: a confirmatory rule “checking account below 0 DM” and an
optional rule “duration above 24 months”, both expected to raise the
default probability. Saga raised 360–361 convergence warnings per fit at
$w_c=10^{-8}$, 66–69 at $10^{-6}$, 3–6 at $10^{-4}$ and none at
$10^{-2}$ or $1$. Test AUC was 0.753–0.756 at $w_c=10^{-8}$ and
0.784–0.797 for all larger values, and fits at $10^{-2}$ took about a
third of the time of fits at $10^{-4}$. The confirmatory coefficient had
the expected positive sign only at $w_c\le 10^{-6}$, in fits that had
not converged; this agreement is an artefact of non-convergence. In
every converged fit the coefficient was negative: the model also retains
the ordinal checking-status variable and tree rules containing the same
condition, which carry the effect, and the sign check flags the
contradiction. The optional rule was retained in every fit. These
results cover one dataset and one split.

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
  data-reference="sec:exp">5</a>); identity across machines is untested
  (Section <a href="#sec:repro" data-reference-type="ref"
  data-reference="sec:repro">4</a>).

- **No sign constraint.** A confirmatory rule is guaranteed to be
  included
  (Definition <a href="#def:inclusion" data-reference-type="ref"
  data-reference="def:inclusion">1</a>), not to keep the direction the
  expert intended. A declared expected sign is checked and a
  contradiction is flagged, but the coefficient is not constrained. On
  German Credit the contradiction occurred in every converged fit
  (Section <a href="#sec:exp" data-reference-type="ref"
  data-reference="sec:exp">5</a>); a sign-constrained, unpenalised
  confirmatory block is the next planned change.

- **Weakly penalised rules can absorb signal.** With a small $w_c$, a
  confirmatory rule correlated with tree rules can take their weight; a
  weak or wrong expert rule then shapes the model. The diagnostics
  report this case; they do not correct it.

- **Calibration not evaluated.** Penalised logistic outputs are often
  too conservative; optional post-hoc calibration exists but reliability
  curves and Brier scores have not been reported.

- **Coarse frequency estimate.** With $B=10$, the 80% threshold means 8
  of 10 draws; the draws are fixed, not resampled.

- **Runtime.** One fit runs about 2,100 saga fits, single-threaded to
  stay within the execution envelope; with the benchmark tolerance it
  took between about 10 minutes (German Credit, 800 training rows) and
  3.5 hours (HMEQ, 4,768 rows). The cost has not been profiled.

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
exactly reproducible, at no loss of AUC on three credit datasets in a
benchmark without expert rules, and includes expert-imposed rules in the
sense of Definition <a href="#def:inclusion" data-reference-type="ref"
data-reference="def:inclusion">1</a>. The effect of expert rules on
discrimination has been examined only in the German Credit sensitivity
study, and their fitted sign can contradict the expert. Its central
empirical claim so far is determinism, not stability under data change.
Future work, in order of priority: sign-constrained, unpenalised
confirmatory rules via bounded optimisation; the perturbation protocol
of Section <a href="#sec:exp" data-reference-type="ref"
data-reference="sec:exp">5</a> with SIRUS as baseline; profiling and
reducing the runtime (e.g. bootstraps in parallel processes, which
preserves determinism) and testing identity across machines; more
bootstrap draws or error-controlled thresholds ; calibration assessment;
and handling of selection bias when labels come only from past audits.

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
