# References — verification status

Rule: no reference enters the paper until its title, authors, venue and year are checked against the publisher or arXiv page, and its content matches how we describe it.

## Already cited in paper v1 (details from `docs/SOTA.md` and memory — volume/pages unchecked)
| Ref | Status | Check |
|---|---|---|
| Friedman & Popescu 2008, AoAS 2(3) — arXiv:0811.1679 | likely correct | pages 916–954 |
| Bénard et al. 2021, EJS 15(1) 427–505 — arXiv:1908.06852 | likely correct | title wording |
| Meinshausen & Bühlmann 2010, JRSS-B 72(4) 417–473 | likely correct | — |
| Shah & Samworth 2013, JRSS-B 75(1) 55–80 | likely correct | pages |
| Zou 2006, JASA 101(476) 1418–1429 | likely correct | pages |
| Nalenz & Augustin 2022, AISTATS, PMLR 151 | from SOTA.md | URL proceedings.mlr.press/v151/nalenz22a |
| Liu & Mazumder 2023, FIRE, KDD — arXiv:2306.07432 | from SOTA.md | exact title |
| Kanamori 2023, LIRE, ECML PKDD — arXiv:2306.11481 | from SOTA.md | — |
| Tan et al. 2025, FIGS, PNAS — doi 10.1073/pnas.2310151122 | from SOTA.md | year/volume |
| Fokkema 2020, `pre`, JSS 92(12) | memory | — |
| Lou et al. 2013, GA2M, KDD | memory | — |
| Nori et al. 2019, InterpretML — arXiv:1909.09223 | from SOTA.md | — |
| Rudin et al. 2024, "Amazing things come from having many good models", ICML | from SOTA.md | authors list |
| Singh et al. 2021, imodels, JOSS 6(61) | from SOTA.md | — |

## Suggested by the AI reviewer — NOT verified, may be inaccurate
| Name in review | What the review says | Action |
|---|---|---|
| HorseRule | rule-structured horseshoe prior, sparser/more stable than L1 | find paper (likely Nalenz & Villani, Bayesian rule ensembles); verify |
| SR4-Fit | SR3 relaxation adapted to RuleFit; Dice–Sørensen stability; RUS, E-IPS readability metrics | search; verify existence and metrics |
| TE2Rules | rule lists extracted from tree ensembles | search; verify |
| IP-based set partitioning | Bonasera & Gualandi, arXiv:2407.00843 (also in SOTA.md) | verify |
| COB | orthogonal gradient boosting for fewer, general rules | search; verify |
| Bayesian stability selection | arXiv:2410.21914 | open arXiv page; verify title/authors |
| CPSS | Shah & Samworth 2013 (already cited) | — |

## Suggested by review 2 — NOT verified
| Name | Claim in review | Action |
|---|---|---|
| BRL (Bayesian Rule Lists) | combinatorial rule learner | likely Letham et al. 2015; verify |
| IMLI | MaxSAT-based interpretable rules | likely Ghosh & Meel; verify |
| gren | group-regularised elastic net with empirical Bayes penalties | likely Münch et al.; verify |
| COB | orthogonal gradient boosting for compact rules | same as review 1; verify |

## Suggested by review 3 — NOT verified
| Name | Claim in review | Action |
|---|---|---|
| RPTE | "strict source disjointness, auditability-first", audit-load measures | find source; high risk of inaccurate description |
| RCProb / RuleCOSI+ | efficient rule extraction from ensembles | verify |
| Node Harvest | rule-based method (likely Meinshausen 2010, AoAS) | verify |
| MNAM | monotonic neural additive models | verify |
| HELOC, GMSC | credit datasets (FICO xML challenge; Kaggle Give Me Some Credit) | check licences |
