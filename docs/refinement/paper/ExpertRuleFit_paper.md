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

Expert RuleFit , introduced in a master’s thesis with an R
implementation, extends RuleFit with confirmatory expert rules and
linear terms that are exempt from penalisation, so that they always
enter the model, and optional ones with an adjustable penalty factor.
The confirmatory/optional distinction, its terminology and the penalty
exemption are theirs; this report uses the same name for a Python
implementation and does not claim them.

#### Contributions.

1.  *Determinism with canonical row order*: internal seeds are fixed,
    the user seed is ignored and the training rows are sorted before
    fitting, so any permutation of the same rows gives the same model;
    the guarantee is stated as an execution envelope with its limits
    (Sections <a href="#sec:method" data-reference-type="ref"
    data-reference="sec:method">3</a>,
    <a href="#sec:repro" data-reference-type="ref"
    data-reference="sec:repro">4</a>).

2.  *A sign-constrained confirmatory block*: confirmatory coefficients
    unpenalised and bounded by the expert’s expected sign in one convex
    problem, with confirmatory inclusion defined precisely
    (Definition <a href="#def:inclusion" data-reference-type="ref"
    data-reference="def:inclusion">1</a>)
    (Section <a href="#sec:method" data-reference-type="ref"
    data-reference="sec:method">3</a>).

3.  *Diagnostics and fingerprints*: sign checks, absorption diagnostics
    and SHA-256 fingerprints of rules, coefficients and predictions in
    an audit bundle.

4.  *A conditioning analysis*: penalty weighting by column rescaling
    makes saga stop far from the optimum without a warning when weights
    are small; standardisation and the weight choice that avoid it. A
    benchmark, an ablation and a perturbation study test these points
    (Section <a href="#sec:exp" data-reference-type="ref"
    data-reference="sec:exp">5</a>).

5.  An open-source implementation built on `imodels` and `scikit-learn`,
    with code, tests, scripts, results and a container recipe at
    <https://github.com/hdjebar/ExpertRuleFit>; datasets are public.
    Table <a href="#tab:results" data-reference-type="ref"
    data-reference="tab:results">1</a> and the perturbation study use
    <span class="smallcaps">ExpertRuleFit</span> 0.8.0 (commit
    `4588f20`, 104 automated tests passing); the expert-rule and
    ablation results of
    Section <a href="#sec:exp" data-reference-type="ref"
    data-reference="sec:exp">5</a> use version 0.7 (commit `ad21d89`).
    Both ran in a container (base digest `07d48240…`, lock file SHA-256
    `990ced9d…`).

# Related work

#### Expert knowledge in rule ensembles.

Expert RuleFit adds expert rules and linear terms to RuleFit’s candidate
set, exempts confirmatory ones from the $\ell_1$/elastic-net penalty
(penalty factor 0) and gives optional ones a common penalty factor; it
is the direct precedent of this work. We keep its confirmatory/optional
design and change how it is fitted and audited: deterministic execution
with canonical row order, sign bounds on the confirmatory block, an
optional selection-frequency filter, and diagnostics.

#### Stable rule ensembles.

SIRUS obtains stability from the forest structure: splits are
discretised on quantiles and the most frequent rules across trees are
kept. Other methods target sparsity: Compressed Rule Ensembles merge
similar rules, FIRE uses a fusion penalty favouring shared antecedents,
LIRE minimises the number of rules needed per prediction, and FIGS grows
a small sum of trees. The `pre` package uses relaxed and adaptive Lasso
to reduce false selections.

#### Stability selection.

Retaining variables selected in most subsample fits is the stability
selection of , with improved error bounds in . Our optional filter
applies the idea with ten fixed draws and no error control.

#### Prior knowledge in penalties.

Reducing the penalty on chosen variables follows the adaptive Lasso and
the use of per-feature penalty factors. Monotonicity and sign
constraints, common in credit scoring, fix the direction of an effect;
our constrained mode applies such bounds to the confirmatory block only.

#### Glass-box models and multiplicity.

Explainable Boosting Machines fit additive shape functions with pairwise
interactions; we use them to propose interaction rules, in a stacked
variant and as a baseline. Many models fit the data almost equally well
; <span class="smallcaps">ExpertRuleFit</span> fixes the draw from this
Rashomon set, which makes the output auditable, not better.

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
These weights drive the cross-validation that picks $\lambda$ and
$\alpha$ (and the optional filter); the default constrained block then
leaves $C$ unpenalised. Coefficients and the intercept are mapped back
to the original scale. Earlier versions used $w_c=10^{-8}$ without
standardisation, which made saga stop far from the optimum without a
warning; $10^{-2}$ is the smallest value that fitted without warnings
(Section <a href="#sec:exp" data-reference-type="ref"
data-reference="sec:exp">5</a>).

<div id="def:inclusion" class="definition">

**Definition 1** (Inclusion). A confirmatory rule $c$ is *included* if
its column is part of the final model and its fitted coefficient is
non-zero ($|\beta_c|>10^{-10}$).

</div>

Inclusion says nothing about the size of $\beta_c$. Confirmatory columns
are never filtered out and are not penalised (below); a rule can still
fail to be included if its column is constant or collinear, if the data
carry no signal for it, or if its declared sign bound is active, and the
estimator then reports it.

<div class="algorithm">

Sort the rows of $(X,y)$ lexicographically (canonical order) Fit
gradient boosting (250 trees, depth 4, seed 42); $R$ = rules kept by
RuleFit’s own $\ell_1$ path (at most 50) Build $Z$ from linear terms,
$R$, $C$, $O$; compute centres $m$, scales $s$ and weights $w$ Fit final
elastic-net `LogisticRegressionCV` on the kept columns of
$(Z-m)/(s\sqrt{w})$ Solve the constrained problem below on the kept
columns ($C$ unpenalised, sign-bounded)

</div>

The user’s `random_state` is ignored (a warning is emitted); all
internal seeds derive from a fixed base seed, and rules must be row-wise
functions of $X$ because the rows are sorted first. Line 2 matters:
imodels’ RuleFit selects its rules with its own $\ell_1$ path before
returning them, so by default our candidates are already a Lasso
selection made without the expert rules; an option takes every distinct
rule of the same trees instead. The optional filter uses half-samples
because the $n$-out-of-$n$ bootstrap of earlier versions put duplicated
rows on both sides of the inner cross-validation; it is off by default
since it reduced stability
(Section <a href="#sec:exp" data-reference-type="ref"
data-reference="sec:exp">5</a>). `DualModel` stacks EBM and
<span class="smallcaps">ExpertRuleFit</span> scores on out-of-fold
predictions; `discover_interaction_rules` lets an EBM propose
interaction rules, which enter as *optional* rules marked
`ebm_discovered`, since confirmatory status is reserved for rules an
expert imposes or has reviewed.

#### Diagnostics.

Per confirmatory rule, the estimator reports its largest correlation
with other candidate columns, its variance inflation factor and the
correlated candidates it leaves inactive, which makes absorption
visible; in the refit mode a declared expected sign $s_c\in\{\pm1\}$ is
only checked.

#### Constrained block (default).

The final step solves one convex problem on the kept columns
$\tilde z_i$ (standardised, not rescaled by $w$):
$$\min_{\beta_0,\beta}\ \frac1n\sum_{i=1}^n \log\bigl(1+e^{-(2y_i-1)(\beta_0+\tilde z_i^\top\beta)}\bigr)+\lambda\sum_{j\notin C} v_j\Bigl(\alpha|\beta_j|+\tfrac{1-\alpha}{2}\beta_j^2\Bigr)
\quad\text{s.t. } s_c\beta_c\ge 0\ \ (c\in C,\ s_c\ \text{declared}),$$
with $v_j=w_o$ for optional rules and 1 otherwise, no penalty on $C$,
and $\lambda=1/(Cn)$ and $\alpha$ from the cross-validation of the refit
mode. The problem is convex; if $\alpha<1$ and $[\mathbf 1, Z_C]$ has
full column rank it is strictly convex and the solution unique even when
tree rules duplicate a confirmatory rule, which motivates a ridge-only
option ($\alpha=0$). We solve it with FISTA and projection onto the
constraints, without randomness. Rules without a declared sign are
unpenalised and free. A confirmatory coefficient held at zero with the
gradient pointing past the bound is reported as *bound-active*: given
the other terms, the data oppose the expert’s sign; it is flagged in the
audit bundle, raises an error with `strict=True`, and the rule is *not*
included in the sense of
Definition <a href="#def:inclusion" data-reference-type="ref"
data-reference="def:inclusion">1</a>. The earlier unpenalised fallback
refit remains available as `confirmatory_mode="refit"`.

# Reproducibility guarantee

The guarantee is bitwise determinism within an execution envelope: the
same set of training rows, with the same column order, on the same
software stack, yields the same rule set and predictions. The envelope
requires pinned versions of Python, NumPy, scikit-learn and imodels;
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
keep the seed whose rules look best (“seed shopping”); the log shows
only that seed. With <span class="smallcaps">ExpertRuleFit</span> the
seed has no effect: for given data, settings and software stack there is
one model, and the remaining choices are settings recorded in the audit
bundle. Row order is a second hidden choice of the same kind, because
boosting subsamples and cross-validation folds index rows by position;
the rows are therefore sorted before fitting, so that only the set of
rows matters (Section <a href="#sec:exp" data-reference-type="ref"
data-reference="sec:exp">5</a>). Deriving internal seeds from a user
seed would bring the free choice back; the fixed draw is still one draw
among many, which is the stability question of
Section <a href="#sec:exp" data-reference-type="ref"
data-reference="sec:exp">5</a>.

The guarantee does not cover different data, a different software stack
or CPU, or column order: in our runs a different column order from
another preprocessing script changed the fitted model, so columns must
be ordered by a fixed schema. Identity across machines has not been
tested: all results in this report come from one x86-64 machine (Intel
Xeon, 2.1 GHz), and runs on another CPU type, even with the same
container, may differ in the last bits of floating-point results.

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
<span class="smallcaps">ExpertRuleFit</span> 0.8.0 is fitted with its
default settings and no expert rules (the RuleFit fits are unchanged
from the previous version of this table, whose code path is untouched).
Since it ignores the seed, it is fitted three times
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
| German Credit |         100         |     1/100     |                     3/3                      | 0.788 ($\pm$<!-- -->0.009) |                    0.794                     |
| Taiwan Credit |         100         |     1/100     |                     3/3                      | 0.764 ($\pm$<!-- -->0.003) |                    0.757                     |
| HMEQ          |         100         |     1/100     |                     3/3                      | 0.903 ($\pm$<!-- -->0.003) |                    0.902                     |

Seed sensitivity (single split, no expert rules). RuleFit: 100 seeds,
mean AUC. <span class="smallcaps">ExpertRuleFit</span>: three
bitwise-identical runs.

</div>

Mean pairwise Jaccard similarity of RuleFit rule sets is 0.045 (German
Credit), 0.058 (Taiwan Credit), 0.036 (HMEQ). The three
<span class="smallcaps">ExpertRuleFit</span> runs per dataset produced
the same rule set (51, 28, 37 terms) and the same probabilities; a fit
took 24 s, 1.6 min and 6.6 min.

Identical output follows from the design: the three runs confirm the
implementation, not stability under data change, and AUC comes from a
single split without confidence intervals.

#### Expert rules, ablation and row order (German Credit, version 0.7).

These runs predate the 0.8.0 defaults (filter on, refit mode); every
configuration gave identical fingerprints in three runs (commands in
`experiments/results/APPENDIX.md`). The expert rules are standard
credit-risk criteria chosen by the author: a confirmatory rule “checking
account below 0 DM” and an optional rule “duration above 24 months”,
both expected to raise the default probability. (i) *Penalty weight.* At
$w_c=10^{-8}$ saga raised about 360 convergence warnings per fit and AUC
fell to 0.753–0.756 (0.784–0.797 from $10^{-4}$); the expected sign
appeared only in non-converged fits. (ii) *Constrained mode.* The refit
gives the confirmatory rule $-1.20$, since the checking-status variable
and tree rules with the same condition carry the effect; the constrained
block holds it at zero, flags it as bound-active, and AUC moves from
0.797 to 0.792 (0.07 s, 755 FISTA iterations). (iii) *Ablation.*
Removing the filter gave AUC 0.789 with 61 instead of 47 terms; removing
expert weighting, 0.784. Fixed-seed RuleFit reached 0.800 in 3 s and EBM
0.803. (iv) *Row order.* Shuffling the training rows changed the rule
set as much as a new RuleFit seed (Jaccard 0.05–0.08, probabilities
differing by up to 0.65); with canonical order all permutations gave
identical output.

#### Stability under perturbation (German Credit).

We refitted each model on 25 stratified 90 % subsamples of the training
split, drawn without replacement with preprocessing refitted inside
each, and compared all pairs of fits on the fixed test set. Exact rule
sets almost never agree (Jaccard 0.003–0.025) because thresholds move;
matching rules one to one by their activations on the test rows
(Hungarian algorithm) gives a semantic similarity of 0.530 for RuleFit
with a fixed seed, 0.527 for
<span class="smallcaps">ExpertRuleFit</span> without the filter and
0.414 with it. Predictions follow the same order: the mean per-row
standard deviation of test probabilities is 0.056, 0.079 and 0.087, and
the share of test rows whose predicted class changes between fits
(ambiguity, adapted from ) is 0.21, 0.38 and 0.40. Mean AUC is 0.785,
0.784 and 0.775. The filter therefore made the model less stable at
about eight times the runtime, and it is off by default since version
0.8.0. <span class="smallcaps">ExpertRuleFit</span> is thus not more
stable than fixed-seed RuleFit here; HMEQ, Taiwan and SIRUS remain to be
run.

# Limitations

- **Stability.** Measured on one dataset only, where
  <span class="smallcaps">ExpertRuleFit</span> was not more stable than
  fixed-seed RuleFit; identity across machines is untested
  (Section <a href="#sec:repro" data-reference-type="ref"
  data-reference="sec:repro">4</a>).

- **Sign bounds can only flag disagreement.** When the data contradict a
  declared sign, as on German Credit, the coefficient is held at zero
  and the rule is not included
  (Definition <a href="#def:inclusion" data-reference-type="ref"
  data-reference="def:inclusion">1</a>).

- **Unpenalised rules can absorb signal.** A confirmatory rule
  correlated with tree rules can take their weight, so a weak or wrong
  expert rule shapes the model; the diagnostics report this, they do not
  correct it.

- **Candidate rules.** By default they are imodels’ own $\ell_1$
  selection, made before expert rules are added; taking all tree rules
  is optional and not yet evaluated. Calibration is not evaluated;
  runtime (eight times longer with the filter) is not profiled.

- **Scope and review.** Binary classification; three credit datasets of
  moderate size; readability not tested with users. Single author,
  internal code review only; the implementation was written with AI
  assistance.

# Application design: tax-audit prioritisation

A tax administration ranking returns for review needs a reason an agent
can state, a selection that can be replayed if contested, and
inspectors’ criteria kept in the model. We sketch a design; the method
has not been applied to tax data.

The target would be an audit outcome (e.g. an adjustment above a
threshold set with the business); tree rules would capture patterns in
past audits; confirmatory rules would encode legal or business criteria
(e.g. deductions above a statutory ceiling); optional rules would hold
inspectors’ hypotheses; and determinism would let a contested selection
be replayed.

Three conditions precede any pilot: *selection bias* (outcomes exist
only for audited files, so a small random audit sample and
inverse-probability weighting are needed); *legal grounds and fairness*
(no variables without legal basis, score distributions compared across
taxpayer groups); and a *human decision* (the score suggests a file, an
agent decides; Article 22 GDPR).

# Conclusion and future work

Building on Expert RuleFit ,
<span class="smallcaps">ExpertRuleFit</span> makes rule-ensemble outputs
exactly reproducible at no loss of AUC and reports when the data
contradict an expert’s sign; it is not more stable under perturbation
than fixed-seed RuleFit. Future work, in order of priority: the
perturbation study on more datasets with SIRUS; candidate rules taken
from the trees rather than from RuleFit’s own $\ell_1$ selection;
separating confirmatory effects from overlapping tree rules; testing
identity across machines; calibration; and selection bias when labels
come only from past audits.

<div class="thebibliography">

99 Bénard, C., Biau, G., da Veiga, S. and Scornet, E. (2021). SIRUS:
Stable and interpretable RUle set for classification. *Electronic
Journal of Statistics*, 15(1), 427–505. Ebner, L. (2021). Expert RuleFit
– Complementing rule ensembles with expert knowledge. Master’s thesis,
Faculty of Science, Vrije Universiteit Amsterdam. R package:
<https://github.com/LuisaEbner/ERF>. Ebner, L., Nalenz, M., ten Teije,
A., van Harmelen, F. and Augustin, T. (2021). Expert RuleFit:
Complementing rule ensembles with expert knowledge. In 19th
International Conference on Artificial Intelligence in Medicine, KR4HC
Workshop. Reprinted in , pp. 117–130. Fokkema, M. (2020). Fitting
prediction rule ensembles with R package pre. *Journal of Statistical
Software*, 92(12). Friedman, J. H. and Popescu, B. E. (2008). Predictive
learning via rule ensembles. *Annals of Applied Statistics*, 2(3),
916–954. Kanamori, K. (2023). Learning locally interpretable rule
ensemble. In *ECML PKDD*. Liu, B. and Mazumder, R. (2023). FIRE: An
optimization approach for fast interpretable rule extraction. In *KDD*.
Lou, Y., Caruana, R., Gehrke, J. and Hooker, G. (2013). Accurate
intelligible models with pairwise interactions. In *KDD*. Marx, C.,
Calmon, F. and Ustun, B. (2020). Predictive multiplicity in
classification. In *ICML*, PMLR 119, 6765–6774. Meinshausen, N. and
Bühlmann, P. (2010). Stability selection. *Journal of the Royal
Statistical Society B*, 72(4), 417–473. Nalenz, M. (2021).
Characterizing model uncertainty in ensemble learning – Towards more
robust representation and learning of tree ensemble methods. Doctoral
dissertation, LMU Munich. <https://edoc.ub.uni-muenchen.de/30379>.
Nalenz, M. and Augustin, T. (2022). Compressed rule ensemble. In
*AISTATS*, PMLR 151. Nori, H., Jenkins, S., Koch, P. and Caruana, R.
(2019). InterpretML: A unified framework for machine learning
interpretability. arXiv:1909.09223. Rudin, C. et al. (2024). Amazing
things come from having many good models. In *ICML*. Shah, R. D. and
Samworth, R. J. (2013). Variable selection with error control: another
look at stability selection. *Journal of the Royal Statistical Society
B*, 75(1), 55–80. Singh, C., Nasseri, K., Tan, Y. S., Tang, T. and Yu,
B. (2021). imodels: a Python package for fitting interpretable models.
*Journal of Open Source Software*, 6(61). Tan, Y. S., Singh, C.,
Nasseri, K., Agarwal, A. and Yu, B. (2025). Fast interpretable
greedy-tree sums. *PNAS*. Zou, H. (2006). The adaptive lasso and its
oracle properties. *Journal of the American Statistical Association*,
101(476), 1418–1429.

</div>
