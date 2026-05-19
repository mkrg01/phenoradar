# Changelog

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
