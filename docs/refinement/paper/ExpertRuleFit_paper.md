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

In regulated settings such as credit, insurance or tax administration
this blocks deployment: a supervisor who re-runs the model must obtain
the same output; a rule given as the reason for a score should not
disappear at retraining for reasons unrelated to the data; and criteria
required by regulation or by experts can be removed by the penalty when
their signal is weak or collinear with other rules.

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

2.  The reproducibility guarantee stated as determinism within an
    execution envelope, with its limits
    (Section <a href="#sec:repro" data-reference-type="ref"
    data-reference="sec:repro">4</a>).

3.  A benchmark on three credit datasets, an ablation, and a protocol
    for stability under data perturbation
    (Section <a href="#sec:exp" data-reference-type="ref"
    data-reference="sec:exp">5</a>).

4.  An open-source implementation built on `imodels` and `scikit-learn`,
    with code, tests, scripts, results and a container recipe at
    <https://github.com/hdjebar/ExpertRuleFit>; datasets are public.
    Table <a href="#tab:results" data-reference-type="ref"
    data-reference="tab:results">1</a> was produced with
    <span class="smallcaps">ExpertRuleFit</span> 0.7.0 at commit
    `281d5a6`, and the German Credit results of
    Section <a href="#sec:exp" data-reference-type="ref"
    data-reference="sec:exp">5</a> at commit `ad21d89`, where default
    settings reproduce the earlier outputs exactly and 93 automated
    tests pass. Both ran in a container built from `python:3.11-slim`
    (digest `07d48240…`) with a hash-locked dependency file (SHA-256
    `990ced9d…`).

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

#### Glass-box models and multiplicity.

Explainable Boosting Machines fit additive shape functions with pairwise
interactions; we use them to propose interaction rules, in a stacked
variant and as a baseline. Many models fit the data almost equally well
: RuleFit’s instability is a random draw from this Rashomon set, and
<span class="smallcaps">ExpertRuleFit</span> fixes the draw, which makes
the output auditable but does not make it better than its neighbours.

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
Coefficients and the intercept are mapped back to the original scale.
Earlier versions used $w_c=10^{-8}$ without standardisation, which made
saga stop far from the optimum without a warning; $10^{-2}$ is the
smallest value that fitted without warnings
(Section <a href="#sec:exp" data-reference-type="ref"
data-reference="sec:exp">5</a>). Inclusion
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
internal seeds derive from a fixed base seed. Post-hoc calibration
(Platt or isotonic) is available but off by default. `DualModel` stacks
EBM and <span class="smallcaps">ExpertRuleFit</span> scores with a
logistic meta-model trained on out-of-fold predictions.
`discover_interaction_rules` uses an EBM to propose interaction rules;
being data-discovered, they enter as *optional* rules with provenance
`source="ebm_discovered"` and must pass the frequency filter.
Confirmatory status is reserved for rules an expert imposes or has
reviewed.

#### Sign check and absorption diagnostics.

A confirmatory rule may declare an expected sign $s_c\in\{+1,-1\}$; in
the refit mode a contradiction is flagged and warned about, not
prevented. Diagnostics report, per confirmatory rule, its largest
correlation with other candidate columns, its variance inflation factor,
and the correlated candidates it leaves inactive, which makes absorption
visible.

#### Constrained mode.

The refit step guarantees inclusion but not direction. With the option
`confirmatory_mode` set to `"constrained"`, the final step instead
solves one convex problem on the stable columns $\tilde z_i$
(standardised, not rescaled by $w$):
$$\min_{\beta_0,\beta}\ \frac1n\sum_{i=1}^n \log\bigl(1+e^{-(2y_i-1)(\beta_0+\tilde z_i^\top\beta)}\bigr)+\lambda\sum_{j\notin C} v_j\Bigl(\alpha|\beta_j|+\tfrac{1-\alpha}{2}\beta_j^2\Bigr)
\quad\text{s.t. } s_c\beta_c\ge 0\ \ (c\in C,\ s_c\ \text{declared}),$$
with $v_j=w_o$ for optional rules and 1 otherwise, no penalty on $C$,
and $\lambda=1/(Cn)$ and $\alpha$ from the cross-validation of the refit
mode. The problem is convex; if $\alpha<1$ and $[\mathbf 1, Z_C]$ has
full column rank it is strictly convex and the solution unique even when
tree rules duplicate a confirmatory rule, which motivates a ridge-only
option ($\alpha=0$). We solve it with FISTA and projection onto the
constraints, without randomness. A confirmatory coefficient held at zero
with the gradient pointing past the bound is reported as *bound-active*:
given the other terms, the data oppose the expert’s sign, and the rule
is not included in the sense of
Definition <a href="#def:inclusion" data-reference-type="ref"
data-reference="def:inclusion">1</a>.

# Reproducibility guarantee

The guarantee is bitwise determinism within an execution envelope: the
same training data, in the same column order, on the same software
stack, yields the same rule set and predictions. The envelope requires
pinned versions of Python, NumPy, scikit-learn and imodels;
single-threaded BLAS and OpenMP; the fixed internal seeds; a validated
input schema with stable feature order; and identical platform settings.
A container recipe (base image pinned by digest, dependencies installed
from a hash-locked file) is provided in the repository.

#### Why not log the seed?

Logging the seed, code and data hash makes one RuleFit run replayable,
but not the choice of that run: in
Table <a href="#tab:results" data-reference-type="ref"
data-reference="tab:results">1</a>, 100 seeds gave 100 rule sets with
nearly the same AUC (standard deviation at most 0.009). A modeller can
keep the seed whose rules look most plausible (“seed shopping”), and the
log shows only the seed that was kept. With
<span class="smallcaps">ExpertRuleFit</span> the seed has no effect: for
given data, settings and software stack there is one model, and the
remaining choices are settings recorded in the audit bundle. Row order
is a second hidden choice of the same kind, because bootstrap draws and
cross-validation folds index rows by position; `row_order="canonical"`
sorts the rows before fitting, so that only the set of rows matters
(Section <a href="#sec:exp" data-reference-type="ref"
data-reference="sec:exp">5</a>). Deriving a logged chain of internal
seeds from a user seed would bring the free choice back, since different
user seeds would give different bootstrap draws; fixing the base seed
makes it part of the method, like the number of draws. The fixed draw is
still one draw among many; how much the result depends on it is the
stability question of
Section <a href="#sec:exp" data-reference-type="ref"
data-reference="sec:exp">5</a>.

The guarantee does not cover different data (adding or correcting rows
may change the rule set), a different software stack or CPU, or
dependence on row order where underlying algorithms break ties by
position. Identity across machines has not been tested: all results in
this report come from one x86-64 machine (Intel Xeon, 2.1 GHz), and runs
on another CPU type, even with the same container, may differ in the
last bits of floating-point results.

# Experiments

#### Data and protocol.

We use German Credit (UCI, 1,000 rows), Taiwan Credit (UCI; a random
subsample of 5,000 of the 30,000 rows, seed 42, to bound runtime) and
HMEQ (5,960 rows). Each dataset is split once into 80% training and 20%
test data (stratified, seed 42). Missing values are imputed (median or
mode), categorical columns ordinally encoded and all columns
standardised, with statistics fitted on the training split. RuleFit
(imodels, 250 trees of depth 4, at most 50 rules) is fitted 100 times
with `random_state` $=0,\dots,99$.
<span class="smallcaps">ExpertRuleFit</span> is fitted with its default
settings and no expert rules, except for the solver tolerance: $10^{-6}$
instead of the default $10^{-4}$, inherited from the original benchmark
and kept so that the rerun changes only the preprocessing and the
software stack. Since it ignores the seed, it is fitted three times
($\texttt{random\_state}=0,1,2$, separate processes) to check that the
runs agree bit for bit (hash of the rule set, SHA-256 of the test
probabilities). A rule set is the set of terms with non-zero
coefficient; Jaccard similarity is averaged over all pairs of runs. All
runs use the pinned software stack (Python 3.11.18, scikit-learn 1.7.2,
NumPy 2.4.6, imodels 3.0.4, single-threaded), in a container built from
the recipe in the repository.

<div id="tab:results">

|               |      Rule sets      |               |                                              |          Test AUC          |                                              |
|:--------------|:-------------------:|:-------------:|:--------------------------------------------:|:--------------------------:|:--------------------------------------------:|
| 2-4(lr)5-6    | RuleFit (100 seeds) |               | <span class="smallcaps">ExpertRuleFit</span> |                            |                                              |
| Dataset       |      distinct       | most frequent |                identical runs                |          RuleFit           | <span class="smallcaps">ExpertRuleFit</span> |
| German Credit |         100         |     1/100     |                     3/3                      | 0.788 ($\pm$<!-- -->0.009) |                    0.790                     |
| Taiwan Credit |         100         |     1/100     |                     3/3                      | 0.764 ($\pm$<!-- -->0.003) |                    0.769                     |
| HMEQ          |         100         |     1/100     |                     3/3                      | 0.903 ($\pm$<!-- -->0.003) |                    0.903                     |

Seed sensitivity (single split, no expert rules). RuleFit: 100 seeds,
mean AUC. <span class="smallcaps">ExpertRuleFit</span>: three
bitwise-identical runs.

</div>

Mean pairwise Jaccard similarity of RuleFit rule sets is 0.045 (German
Credit), 0.058 (Taiwan Credit), 0.036 (HMEQ). The three
<span class="smallcaps">ExpertRuleFit</span> runs per dataset produced
the same rule set (43, 42, 48 terms) and the same probabilities.

#### What the results show and do not show.

Table <a href="#tab:results" data-reference-type="ref"
data-reference="tab:results">1</a> uses no expert rules. RuleFit’s
output depends heavily on the seed;
<span class="smallcaps">ExpertRuleFit</span> removes this dependence
without loss of discrimination. Identical output follows from the
design; the three runs confirm the implementation, not stability under
data change. AUC comes from a single split without confidence intervals.

#### Expert rules, ablation and row order (German Credit).

With the library tolerance $10^{-4}$, every
<span class="smallcaps">ExpertRuleFit</span> configuration gave
identical fingerprints in three runs (tables and commands in the
repository file `experiments/results/APPENDIX.md`). The expert rules are
standard credit-risk criteria chosen by the author: a confirmatory rule
“checking account below 0 DM” and an optional rule “duration above 24
months”, both expected to raise the default probability. (i) *Penalty
weight.* Over $w_c\in\{10^{-8},\dots,1\}$ and $w_o\in\{0.1,0.3,1\}$,
saga raised 360–361 convergence warnings per fit at $w_c=10^{-8}$, 3–6
at $10^{-4}$ and none from $10^{-2}$; AUC was 0.753–0.756 at $10^{-8}$
and 0.784–0.797 otherwise. The confirmatory coefficient had the expected
sign only in non-converged fits ($w_c\le10^{-6}$). (ii) *Constrained
mode.* In the refit mode the coefficient is $-1.20$ (the ordinal
checking-status variable and tree rules with the same condition carry
the effect). In the constrained mode it is held at zero, bound-active,
and reported as such; AUC moves from 0.797 to 0.792 (0.785 with ridge),
and the solve takes 0.07 s after 755 FISTA iterations. (iii) *Ablation.*
Without the frequency filter, AUC is 0.789 with 61 instead of 47 terms;
without expert weighting, 0.784; without the refit step, unchanged (the
refit was not triggered). RuleFit with a fixed internal seed reached
0.800 with 40 terms in 3 s, against 233 s for
<span class="smallcaps">ExpertRuleFit</span>, of which 90 % is the
bootstrap filter; EBM reached 0.803 using all 80 terms for every case.
On this split the frequency filter adds neither accuracy nor
determinism; its value, if any, is stability under data perturbation,
which the next protocol tests. (iv) *Row order.* Shuffling the training
rows changed the rule set as much as a new RuleFit seed (Jaccard
0.05–0.08 against the original order, probabilities differing by up to
0.65); with `row_order="canonical"` all permutations gave
bitwise-identical output.

#### Perturbation protocol (not yet run).

100 bootstrap and 100 90%-subsamples of the training set; fixed-seed
RuleFit, <span class="smallcaps">ExpertRuleFit</span>, SIRUS and
$\ell_2$ logistic regression on each; pairwise Jaccard and per-rule
selection frequencies as in ; AUC with confidence intervals over 20
repeated splits.

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
  data-reference="sec:exp">5</a>). The optional constrained mode
  enforces the sign, at the price of a bound-active, zero coefficient
  when the data disagree; it is not yet the default.

- **Weakly penalised rules can absorb signal.** With a small $w_c$, a
  confirmatory rule correlated with tree rules can take their weight; a
  weak or wrong expert rule then shapes the model. The diagnostics
  report this case; they do not correct it.

- **Untested details.** Calibration (reliability curves) has not been
  evaluated; with $B=10$ the 80% threshold means 8 of 10 fixed draws,
  without error control.

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

| Model element               | Tax-audit counterpart                                                                                                                |
|:----------------------------|:-------------------------------------------------------------------------------------------------------------------------------------|
| Target $y$                  | Audit outcome, e.g. adjustment above a threshold set with the business                                                               |
| Tree-derived rules          | Patterns in past audit results, e.g. high deductions combined with mainly salaried income                                            |
| Confirmatory rules          | Legal or business criteria, e.g. a gap between declared salary and the employer’s statement, or deductions above a statutory ceiling |
| Optional rules; determinism | Inspectors’ hypotheses, kept only if supported by data; a contested selection can be replayed                                        |

Mapping of model elements to a tax-audit setting (illustrative).

Three conditions precede any pilot: *selection bias* (outcomes exist
only for audited files, so a small random audit sample and
inverse-probability weighting are needed); *legal grounds and fairness*
(no variables without legal basis, score distributions compared across
taxpayer groups); and a *human decision* (the score suggests a file, an
agent decides; Article 22 GDPR). A pilot would compare the hit rate per
audit hour with the current method on the random sample.

# Conclusion and future work

<span class="smallcaps">ExpertRuleFit</span> makes rule-ensemble outputs
exactly reproducible, at no loss of AUC on three credit datasets in a
benchmark without expert rules, and includes expert-imposed rules in the
sense of Definition <a href="#def:inclusion" data-reference-type="ref"
data-reference="def:inclusion">1</a>. The effect of expert rules has
been examined only on German Credit; there, the expert’s sign was
contradicted by the data, and the constrained mode makes that
contradiction explicit instead of reporting a coefficient of the wrong
sign. Its central empirical claim so far is determinism, not stability
under data change. A fixed-seed RuleFit matched its accuracy on German
Credit, so the case for the frequency filter now rests on stability
under perturbation. Future work, in order of priority: the perturbation
protocol of Section <a href="#sec:exp" data-reference-type="ref"
data-reference="sec:exp">5</a> with SIRUS and fixed-seed RuleFit as
baselines; evaluation of the constrained mode on more datasets;
profiling and reducing the runtime (e.g. bootstraps in parallel
processes, which preserves determinism) and testing identity across
machines; more bootstrap draws or error-controlled thresholds ;
calibration assessment; and handling of selection bias when labels come
only from past audits.

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
