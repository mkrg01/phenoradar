# Changelog

## Unreleased

### Breaking Changes

* replace scikit-learn logistic elastic net with glum's weighted binomial GLM;
  replace logistic `C` with native `alpha`, add `gradient_tol`, and remove
  `model.logistic_solver`; use defaults `alpha=0.01`, `l1_ratio=0.5`,
  `max_iter=100`, and `gradient_tol=1e-6`
* fit optional logistic warm-start paths in descending `alpha` order and prefer
  larger `alpha` under one-SE selection; regenerate previous logistic configs
  and model bundles for the new backend

### Features

* reuse stronger candidate lambdas as native paths for selected logistic models
  in outer CV and final refit, preserving the exact selected lambda and fitting
  each path on that refit's own training data
* enable logistic grid-search warm starts by default, configurable with
  `model.logistic_warm_start_path`; parallelize independent fold/parameter
  paths while fitting each alpha path from strongest to weakest regularization
* add optional `split.require_both_labels_per_group` filtering before CV,
  routing single-label groups to external test and keeping them out of CV
  and final-refit training; support family-level eligibility from existing
  metadata annotations without rewriting holdout columns
* generate all-species and `_accepted_only` prediction/evaluation SVGs when
  abstention is enabled, with recomputed selective metrics and explicit messages
  for empty or single-label accepted populations; preserve publication layouts
  without added population subtitles
* add `sparse_feature_filter.scope` with `all_samples`, `any_trait`, `trait_0`,
  and `trait_1` populations; support label-independent pooled nonzero fractions
  and read legacy `within_trait` settings as the corresponding explicit scope
* add observed-only standardization and neutral missing-expression inputs for
  logistic regression, with optional zero-as-missing handling and fixed 0.8
  coefficient-weight coverage abstention; persist policies in version 3 model
  bundles and report selective decisions, missing evidence, and accepted-only
  evaluation alongside raw predictions
* add one candidate-evidence-style PDF per misclassified OOF species, with
  held-out-fold ensemble probabilities, signed local linear contributions, and
  expression references restricted to species sampled for that fold's training
* allow random-forest runs to preserve absent orthogroup coordinates as `NA`
  with `preprocess.absent_feature_fill=nan`, including model-bundle prediction
* add trait-targeted sparse filtering and direction-aware ranked filtering so
  models can retain only features with higher train-fold expression in trait 1
  (or symmetrically trait 0)
* add final-refit feature-importance, signed-coefficient, feature-filter, and
  model-selection artifacts under `model/`, plus external-test top-feature
  expression grouped by final-model confusion status
* add a top-feature `log2(TPM + 1)` small-multiple view with species points grouped by
  OOF TP, FN, TN, and FP status
* allow single-label outer-CV validation groups while keeping two-label training folds mandatory;
  emit undefined two-class fold metrics as `NA`, write split diagnostics for every fold, and add
  reproducible `stratified_group_kfold` for outer and inner CV
* add optional group-level bootstrap confidence intervals for pooled OOF metrics,
  with auditable replicate tables and an interval figure
* add monotonic stage/fold/sample-set/candidate timing traces and top-level run
  timing metadata for performance profiling

### Bug Fixes

* omit the internal group-subsampling repeat index from config templates so
  increasing `group_subsample_repeats` works without removing internal fields
* preserve the complete pre-transform feature schema in model bundles so sample-rank predictions
  are invariant to unrelated input features

### Performance

* reuse row-local expression transforms across inner-CV folds and vectorize pair-aware feature
  ordering; expose inner-CV preprocessing as a dedicated timing stage
* scan and normalize outer-CV expression rows once through a temporary Parquet cache, then build
  dense matrices by integer-coordinate accumulation instead of a wide dataframe pivot
* parse orthogroup annotations only for features used in figures, reuse validated top-feature
  expression values for tree heatmaps, and share one nonzero mask across trait-level sparse counts

## [0.5.0](https://github.com/mkrg01/phenoradar/compare/v0.4.0...v0.5.0) (2026-09-16)


### Features

* add candidate-level evidence reports for positive predictions ([62d62a9](https://github.com/mkrg01/phenoradar/commit/62d62a994aed0eef194f3629dec4ea1ce88049aa))
* add config sweeps and ranked feature filter comparisons ([c29569f](https://github.com/mkrg01/phenoradar/commit/c29569fa01bf44ae89320bc8fb570e36c859ce4e))
* add convergence diagnostics and logistic solver options ([b1c8a7d](https://github.com/mkrg01/phenoradar/commit/b1c8a7da59fb8a645b76052c175baae59ddf213e))
* add CV vs external metric comparison figure ([b774642](https://github.com/mkrg01/phenoradar/commit/b7746429be4ad4edd44b323274c635d87503e7ba))
* add detailed pipeline timing instrumentation ([4605414](https://github.com/mkrg01/phenoradar/commit/4605414bb546295c6a60ef427e2edc9e64290670))
* add explicit scopes for sparse feature filtering ([985403a](https://github.com/mkrg01/phenoradar/commit/985403a728fd512b11853e29128a8dafc5854b1c))
* add external test validation figures ([aebaa00](https://github.com/mkrg01/phenoradar/commit/aebaa00b38cd74064a946978e7a9e98dafe59545))
* add final-refit model interpretation artifacts ([a636249](https://github.com/mkrg01/phenoradar/commit/a636249f632395a2444f440908a226cb8fdf3a84))
* add fold-wise feature importance heatmap figure ([8e1abca](https://github.com/mkrg01/phenoradar/commit/8e1abca80abee480c24b0373bf66c8fe70c44239))
* add full-run inference probability distribution figure ([818b023](https://github.com/mkrg01/phenoradar/commit/818b023e0c338d3959ed7a322fb954a0ff173fa8))
* add group bootstrap CIs for OOF metrics ([ee7ed71](https://github.com/mkrg01/phenoradar/commit/ee7ed71225ffdc9c9df321961ee93215dd5b1053))
* add one-SE model selection and diagnostic curve ([b7426e1](https://github.com/mkrg01/phenoradar/commit/b7426e1668871381b0f976771644189d5af092f9))
* add orthogroup annotation labels ([3d76526](https://github.com/mkrg01/phenoradar/commit/3d76526902e9d47357b4bbbdef6c8f43fff58bed))
* add outer-CV feature stability diagnostics ([c565667](https://github.com/mkrg01/phenoradar/commit/c5656677e7b44f59ca557be1e8a5136c568532d6))
* add per-species CV misclassification evidence ([c257f20](https://github.com/mkrg01/phenoradar/commit/c257f20e7bc8cd25c53cb8141388e135f08cc318))
* add selected feature count by fold ([f3b691c](https://github.com/mkrg01/phenoradar/commit/f3b691cfd84c968c59e81519cfc632ca65039e53))
* add stratified group CV and single-label fold support ([11cbf10](https://github.com/mkrg01/phenoradar/commit/11cbf109e2c3107e41ca3cb35bb5c9d715d03ac4))
* add taxon group summaries and probability figures ([7f9b73c](https://github.com/mkrg01/phenoradar/commit/7f9b73c1b17d7d2dca7ce34bdccf893959e5aa08))
* add top-feature expression plots by confusion group ([f06ad71](https://github.com/mkrg01/phenoradar/commit/f06ad715c07abcdd924b4debba28988dcee6a7c5))
* add training-group count sensitivity experiments ([7c1f1ac](https://github.com/mkrg01/phenoradar/commit/7c1f1aca5f1dfec4c129d89a2d1d2d5dfe56093a))
* add trait-specific feature filtering ([5b7b692](https://github.com/mkrg01/phenoradar/commit/5b7b692baa590f92c8967360c21003187138bb84))
* allow pair-aware filtering with taxonomic ran ([3e3e650](https://github.com/mkrg01/phenoradar/commit/3e3e650f62c4d78bcb1947420dc429051aa6c5f7))
* change low prevalence filter to trait-aware sparse feature filter ([e552370](https://github.com/mkrg01/phenoradar/commit/e5523700f0b2318c5f4f92206f7c5c96273a54cf))
* color tree heatmap species labels by confusion group ([2296e57](https://github.com/mkrg01/phenoradar/commit/2296e575a3ea6016bdf15af9b2c724b6bad0899f))
* enable group-bootstrap evaluation workflow ([e2f35f6](https://github.com/mkrg01/phenoradar/commit/e2f35f6939f7943881108e90c6814241c369a029))
* expand model evaluation and migrate elastic net to glum ([e005ce1](https://github.com/mkrg01/phenoradar/commit/e005ce1bbe389301b99e40084ffcadc3e92e7852))
* generate all-species and accepted-only figures for abstention ([e662a27](https://github.com/mkrg01/phenoradar/commit/e662a27e9ab00c65bf72e311614c7af5c4db6ff7))
* migrate elastic net logistic regression to glum ([a407638](https://github.com/mkrg01/phenoradar/commit/a40763841059981c16343dabca953b8852ae8b72))
* neutralize missing expression and add fixed abstention ([89a987f](https://github.com/mkrg01/phenoradar/commit/89a987f5dc1937716fc3d7a0f6fbdb351afc959a))
* optionally restrict CV to groups containing both labels ([539fad3](https://github.com/mkrg01/phenoradar/commit/539fad3920a4f1c4c02f176e1fb3d29b88da2471))
* organize output artifacts by stage directories ([3beff1e](https://github.com/mkrg01/phenoradar/commit/3beff1e97bcaa750b0cc548bc5df6ef05213a156))
* support NA filling for absent orthogroups in random forests ([420cc10](https://github.com/mkrg01/phenoradar/commit/420cc103d5f5fd82c6fdb9b39285b5e108e65ac4))


### Bug Fixes

* add annotated orthogroup figure variants ([6c647dc](https://github.com/mkrg01/phenoradar/commit/6c647dcac3cbaadbb92ad9331b64d3191f35d6ea))
* add configurable top feature count for figures ([801baa4](https://github.com/mkrg01/phenoradar/commit/801baa410e56f8be32de753e3827b39320dd7fea))
* add white background to tree SVG outputs ([160db0d](https://github.com/mkrg01/phenoradar/commit/160db0d101c3063765e0a1031ac083d3db28e665))
* align vertical tree heatmap labels ([0bc9a6f](https://github.com/mkrg01/phenoradar/commit/0bc9a6fa0dae76de15d0aea97aece65517e426ee))
* bundle c4_tiny dataset with the package ([8788083](https://github.com/mkrg01/phenoradar/commit/8788083260e2f77b31384cfa241a5acf351fc0ef))
* correct y axis labels ([47188a7](https://github.com/mkrg01/phenoradar/commit/47188a73710c64881b49c7d97c459c1376391ccf))
* define schema for CV valid fold counts ([46b8756](https://github.com/mkrg01/phenoradar/commit/46b8756340f667cf8512628db863140e5d3c89e0))
* fix slow feature membership check in matrix pivot ([150aed0](https://github.com/mkrg01/phenoradar/commit/150aed08265f80fa7a7aefa3967b937b04e4b595))
* fix threshold selection curve axis and legend ([9199076](https://github.com/mkrg01/phenoradar/commit/9199076fdf793a8afab369cebefa61136cc7efc0))
* format feature filter funnel stage labels ([6d53222](https://github.com/mkrg01/phenoradar/commit/6d53222188d1885c0b14dc633ce2416f5525c647))
* guard report rankings with experiment fingerprints ([5da7d40](https://github.com/mkrg01/phenoradar/commit/5da7d401c056f8414bcc2dedd3141000bd592059))
* improve condition metrics figure readability ([1aead0b](https://github.com/mkrg01/phenoradar/commit/1aead0be250cbe28eff19988a5a1444e698c8ce8))
* improve feature filter funnel summary display ([d8bffb4](https://github.com/mkrg01/phenoradar/commit/d8bffb4a643ac8cb0e2f8c3cd6ad2e4fe22c382f))
* improve feature heatmap annotations and legend ([4a8541b](https://github.com/mkrg01/phenoradar/commit/4a8541bb2da2c3c4763f86f77a1fb8c9f96bee69))
* improve figure label and legend spacing ([f43f497](https://github.com/mkrg01/phenoradar/commit/f43f4977dcd64b5d353b5de5eb77cd3dc3026957))
* improve figures and feature filter funnel stage display ([ecc7ccf](https://github.com/mkrg01/phenoradar/commit/ecc7ccfd30d2e498d5dda5df8b2d0f4997a49b13))
* improve fold separation in CV trait probability plot ([997307d](https://github.com/mkrg01/phenoradar/commit/997307db31c38cba916975a51299d9cf4097bc99))
* improve tree_prediction_cv.svg readability and group labels ([7508206](https://github.com/mkrg01/phenoradar/commit/7508206fd368bc012a323a3776c19ade807c15ba))
* label annotations before orthogroup ids ([f70d3fd](https://github.com/mkrg01/phenoradar/commit/f70d3fddaea2e3797b6b1f126c15a6285605b492))
* label CV loss axis as Log loss ([0d12a5e](https://github.com/mkrg01/phenoradar/commit/0d12a5ef773a6033bd34aa6581659454749aa845))
* make evaluation metric and threshold contracts explicit ([95a41d4](https://github.com/mkrg01/phenoradar/commit/95a41d483dc6fc2a7d14a415f6034f9a7220ba00))
* make training group count exact and improve sensitivity plots ([b048189](https://github.com/mkrg01/phenoradar/commit/b048189373f97317db1b7fcc79f317160b86b00a))
* make training group subsampling repeats count-based ([58c7873](https://github.com/mkrg01/phenoradar/commit/58c78733f30e0806d811ba1f12f8d7bdc81d3cd2))
* parallelize run artifact figure generation ([6758453](https://github.com/mkrg01/phenoradar/commit/67584531c9ea90f0db8474c0deb804a862101d91))
* place metric x-axes at zero baseline ([4f64ce3](https://github.com/mkrg01/phenoradar/commit/4f64ce3243e3f996f4c03a004961a85e4b4bc616))
* preserve plot width for annotated labels ([56a9d38](https://github.com/mkrg01/phenoradar/commit/56a9d381509033a99aa5f123176185802b6b6636))
* preserve PR curve plotting order ([b04411e](https://github.com/mkrg01/phenoradar/commit/b04411e4e098ce553c0f5433ac72280687f55599))
* preserve raw feature schema for rank transforms ([8539424](https://github.com/mkrg01/phenoradar/commit/85394249af0baeccee7ea1d104cbb24f76027cb7))
* reduce outer-CV inference memory usage ([95e19b1](https://github.com/mkrg01/phenoradar/commit/95e19b11d79b6f4c1ca2f0cb74c8ee3a5ed46b03))
* refine coefficients figure ([f6e7b6e](https://github.com/mkrg01/phenoradar/commit/f6e7b6e442bd66102ccb3fe5525201cf09bd98dd))
* refine coefficients signed top figure layout and labels ([3b869cc](https://github.com/mkrg01/phenoradar/commit/3b869cca8a43e8b1d98541c985e786bf88c272e3))
* refine condition metrics figure axes and styling ([8610473](https://github.com/mkrg01/phenoradar/commit/861047346f217f42207f8819435b2b59a734c291))
* reject invalid TPM values and enforce sample weighting ([6dfa5e1](https://github.com/mkrg01/phenoradar/commit/6dfa5e1575a5a5d1ac10f31de9eafd0ed9ebc536))
* remove CV-derived threshold selection ([215f95b](https://github.com/mkrg01/phenoradar/commit/215f95b14d988ccdc24359b848da274e206988ed))
* remove metadata tree rank override ([c6680cd](https://github.com/mkrg01/phenoradar/commit/c6680cd69f3f8501caaf6dae8e674aa55c7acc6a))
* remove model_sparsity_scatter.svg ([cc3623b](https://github.com/mkrg01/phenoradar/commit/cc3623b9b6e8559d2c99105a294c4c8132e95a9c))
* remove pairwise improvement figures from multi-condition studies ([85fab64](https://github.com/mkrg01/phenoradar/commit/85fab64ac1092264097bdcc16f40f3ae94df55a9))
* remove selected-features-by-fold preprocessing SVG output ([00b6c86](https://github.com/mkrg01/phenoradar/commit/00b6c860a388eb9619cd296b31749e1a07d4756f))
* rename tree contrast pair SVG to tree_group ([42fbb65](https://github.com/mkrg01/phenoradar/commit/42fbb658cff4d6673f2ef35cd43a09f4e9b29ae9))
* render probability histograms with contiguous bins ([3b5fc69](https://github.com/mkrg01/phenoradar/commit/3b5fc69fcfc0c74e5a08dbac2a6169ecbb6cf264))
* respect metric direction in report rankings ([23edc6c](https://github.com/mkrg01/phenoradar/commit/23edc6ced1604a7f3a7d34fa9faeb776d845343d))
* scope build provenance to the PhenoRadar installation ([047d8d4](https://github.com/mkrg01/phenoradar/commit/047d8d48c56325d1b348dba510958be8d6cb3f70))
* show final refit loss x-axis label ([29301e5](https://github.com/mkrg01/phenoradar/commit/29301e5915524fd161366270d64e2d77b921af68))
* simplify metadata inputs and expose all configuration options ([e3b80c2](https://github.com/mkrg01/phenoradar/commit/e3b80c2a0084e162d805bd8578635c4fc18e907f))
* skip redundant max_features conditions for none filter ([98c6d19](https://github.com/mkrg01/phenoradar/commit/98c6d19c8a9cb6b0bdf6d0ce0325c0a969df204b))
* split CV ROC and PR curve figures ([1d256e0](https://github.com/mkrg01/phenoradar/commit/1d256e06a8e49f9688e1ef4c2888d00c97c641c5))
* split feature filter funnel by scope ([f41c014](https://github.com/mkrg01/phenoradar/commit/f41c014dccb87f421f19bf390e40f79ddead02a2))
* update feature filter funnel labeling ([a03c280](https://github.com/mkrg01/phenoradar/commit/a03c280182ffb18d1dddbd0ffc934b4d34cc888a))
* use 1-based fold identifiers ([77b6b61](https://github.com/mkrg01/phenoradar/commit/77b6b61217a789e6a01655dc9ddd98192cf02a76))
* use annotations in standard outputs ([b88a7e5](https://github.com/mkrg01/phenoradar/commit/b88a7e571ab229f6c1710214c30a14029ce9d854))
* use id-first labels for annotated heatmaps ([aed9bf7](https://github.com/mkrg01/phenoradar/commit/aed9bf710356ea1632962ec31dac85348c8182c6))
* use sentence case for one-SE log loss label ([554e347](https://github.com/mkrg01/phenoradar/commit/554e347756c186abbd4499f4615a3414b13c4af9))
* use square panels for CV ROC and PR curves ([6aa4ed2](https://github.com/mkrg01/phenoradar/commit/6aa4ed23ca8c39183a97a32858af31ea0710fd44))


### Performance Improvements

* accelerate CV preprocessing and expression matrix construction ([23785c4](https://github.com/mkrg01/phenoradar/commit/23785c437398dbd90ee044bb81011638c18030a4))
* reduce figure and inner-CV preprocessing overhead ([85bcfb5](https://github.com/mkrg01/phenoradar/commit/85bcfb5a525253ac27006008e4b8f758cc2fd6a0))
* reduce repeated TPM scans during run preparation ([7b26e9e](https://github.com/mkrg01/phenoradar/commit/7b26e9e17edc2ab0bdb9a5af4b521a0cb0ecaa47))


### Dependencies

* install tree libraries by default ([cd6d248](https://github.com/mkrg01/phenoradar/commit/cd6d248eed3cad334069b7f8b4abfde506f68e5d))


### Documentation

* annotate available choices in sample config ([878b301](https://github.com/mkrg01/phenoradar/commit/878b301dee66b226d64a11e9a121356f199f8587))

## [0.4.0](https://github.com/mkrg01/phenoradar/compare/v0.3.0...v0.4.0) (2026-05-19)


### Features

* add automatic species taxid generation ([173725e](https://github.com/mkrg01/phenoradar/commit/173725e2544d8b105d1727d482212d9fa9f79f23))
* add configurable expression transforms and feature scaling ([8f69178](https://github.com/mkrg01/phenoradar/commit/8f69178d2857b074964949a5730e7ecad2228b1e))
* add contrast-pair tree visualization ([910599d](https://github.com/mkrg01/phenoradar/commit/910599d5ce4af455b3693ef5061aad53ceb18a50))
* add explicit holdout and taxonomic rank blocking ([1fc142b](https://github.com/mkrg01/phenoradar/commit/1fc142be34cdaa3f9594580502e54b6fd232af43))
* add fold-to-validation-group mapping artifact ([a267610](https://github.com/mkrg01/phenoradar/commit/a26761082e408dc79e4a55eb37ae5df270e6a4e9))
* add NCBI tree-backed metadata generation ([af70b4c](https://github.com/mkrg01/phenoradar/commit/af70b4c0221dda38f4c8b4ca933cda2895e662c2))
* add pair-aware feature filtering to preprocessing ([06e9db9](https://github.com/mkrg01/phenoradar/commit/06e9db9a467e6e6f9d94480ac9facee140f9441f))
* add retained-feature artifacts and fold heatmap ([bdcdf91](https://github.com/mkrg01/phenoradar/commit/bdcdf91625847c8cf48e7dfa4d66f0b54e2b0c22))
* add tree feature heatmap artifacts ([22b6c31](https://github.com/mkrg01/phenoradar/commit/22b6c31310ba3e12186899c352ec2c23b178acbf))
* add tree prediction annotations and Toytree figures ([f34b953](https://github.com/mkrg01/phenoradar/commit/f34b9535d6025e060bfa4ef4a4b36f0832192679))
* cache inner-CV preprocessing for candidate scoring ([178b78b](https://github.com/mkrg01/phenoradar/commit/178b78b69eb15e1efb7b12904902785fc8eb4afa))
* show fold-level interpretation variation ([e0038c6](https://github.com/mkrg01/phenoradar/commit/e0038c61bddb689292ba86943c40e70a98a50ae4))


### Bug Fixes

* require scikit-learn 1.8+ for logistic l1_ratio semantics ([4210253](https://github.com/mkrg01/phenoradar/commit/421025354e6b2109b67bfd39bededb7e3228adc2))
* run model selection per sampled set instead of mixing hyperparameter and species-set combinations ([8cf248f](https://github.com/mkrg01/phenoradar/commit/8cf248f0ed1e9da259f4a4a5eca9bbe122ce0126))
* standardize model selection ranges on end/end_exp ([5cec7eb](https://github.com/mkrg01/phenoradar/commit/5cec7ebebe82077b6bb98b4a252addd7297e8d5e))
* type optional ete4 taxonomy integration for mypy ([d88e575](https://github.com/mkrg01/phenoradar/commit/d88e5755854bbee70e1dde9be2f6b7a04fe3af44))


### Documentation

* clarify model selection search space range semantics ([cc56b66](https://github.com/mkrg01/phenoradar/commit/cc56b664f24f9d434d026d37217ef2b17f971b4c))

## [0.3.0](https://github.com/mkrg01/phenoradar/compare/v0.2.0...v0.3.0) (2026-03-11)


### Features

* add feature-filter/sparsity artifacts and visualization ([a05b6ac](https://github.com/mkrg01/phenoradar/commit/a05b6ac937aa7023d0091fb4a026400d4c5593b4))
* add feature-filter/sparsity artifacts and visualization ([8e423c5](https://github.com/mkrg01/phenoradar/commit/8e423c52587cdbd92497cc62a639ab6dab864717))


### Documentation

* remove release automation documentation ([98029bb](https://github.com/mkrg01/phenoradar/commit/98029bb6c3bf2ed532dddadb9dadf947c667ee2c))

## [0.2.0](https://github.com/mkrg01/phenoradar/compare/v0.1.0...v0.2.0) (2026-03-10)


### Features

* add CV train/validation log-loss artifact and visualization ([f86083f](https://github.com/mkrg01/phenoradar/commit/f86083fd444b5831eb3c22f298726ffda5a9d05e))
* **figures:** add model selection trials panel visualization ([45f2929](https://github.com/mkrg01/phenoradar/commit/45f2929208b1a374608288fac920535487b2a73e))
* **figures:** add species-level CV/external probability plots ([281d99e](https://github.com/mkrg01/phenoradar/commit/281d99e9c5b804ae1ddcd561d9921f908d1512f4))
* make log_loss the default selection metric and add split-level loss diagnostics/figures ([e54a0f5](https://github.com/mkrg01/phenoradar/commit/e54a0f585f4e392ae16f3dabaf032ce9daf0509d))
* **model-selection:** add log_loss as selectable metric and make it the default ([ffc1524](https://github.com/mkrg01/phenoradar/commit/ffc15243e51da35ce26cdfd947abee057aa1fa90))
* **run:** add split-level loss artifacts and figures for final refit ([a9b9476](https://github.com/mkrg01/phenoradar/commit/a9b9476c834c0671ce5c6963b0f2512d3bd22333))


### Documentation

* add active-development warning callout ([32461e0](https://github.com/mkrg01/phenoradar/commit/32461e041d68e324e0edd4e7aebbd285e6787275))

## 0.1.0 (2026-02-28)


### Features

* PhenoRadar baseline ([0081f23](https://github.com/mkrg01/phenoradar/commit/0081f2384339810ec449288c5532dad5ad104338))


### Documentation

* update Python badge for supported versions ([c9f9417](https://github.com/mkrg01/phenoradar/commit/c9f9417516f4b840d029460828f000ea6977ccf8))
