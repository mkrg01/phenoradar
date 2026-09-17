"""Pydantic schema for configuration validation."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeFloat,
    PositiveFloat,
    PositiveInt,
    field_validator,
    model_validator,
)

ExecutionStage = Literal["cv_only", "full_run"]
OuterCvStrategy = Literal["logo", "group_kfold", "stratified_group_kfold"]
SamplingStrategy = Literal["all_samples", "group_balanced"]
WeightingMode = Literal["none", "group_label_inverse"]
ModelName = Literal["logistic_elasticnet", "linear_svm", "random_forest"]
ProbabilityAggregation = Literal["mean", "median"]
SearchStrategy = Literal["grid", "random", "tpe"]
CandidateSourcePolicy = Literal["per_sample_set", "reuse_first_sample_set"]
SelectionMetricName = Literal["mcc", "balanced_accuracy", "log_loss"]
SelectionRule = Literal["best", "one_se"]
CorrelationMethod = Literal["pearson", "spearman"]
ExpressionTransformMethod = Literal["none", "log1p", "sample_rank", "sample_percentile_rank"]
FeatureScalingMethod = Literal["none", "standard"]
RankedFeatureFilterMethod = Literal["none", "pair_aware", "unpaired", "variance"]
AbsentFeatureFill = Literal[0, "nan"]
SparseFeatureScope = Literal["all_samples", "any_trait", "trait_0", "trait_1"]


class StrictModel(BaseModel):
    """Base model that rejects unknown configuration keys."""

    model_config = ConfigDict(extra="forbid")


class DiscreteRangeSpec(StrictModel):
    """Discrete floating-point range generator."""

    type: Literal["range"]
    start: float
    end: float
    step: float = Field(gt=0)
    inclusive_end: bool = False

    @model_validator(mode="after")
    def validate_bounds(self) -> DiscreteRangeSpec:
        if self.end < self.start:
            raise ValueError("range requires end >= start")
        return self


class IntRangeSpec(StrictModel):
    """Discrete integer range generator."""

    type: Literal["int_range"]
    start: int
    end: int
    step: int = Field(gt=0)
    inclusive_end: bool = False

    @model_validator(mode="after")
    def validate_bounds(self) -> IntRangeSpec:
        if self.end < self.start:
            raise ValueError("int_range requires end >= start")
        return self


class LogRangeSpec(StrictModel):
    """Discrete logarithmic range generator."""

    type: Literal["log_range"]
    base: float
    start_exp: float
    end_exp: float
    step_exp: float = Field(gt=0)
    inclusive_end: bool = False

    @model_validator(mode="after")
    def validate_bounds(self) -> LogRangeSpec:
        if self.base <= 0 or self.base == 1:
            raise ValueError("log_range requires base > 0 and base != 1")
        if self.end_exp < self.start_exp:
            raise ValueError("log_range requires end_exp >= start_exp")
        return self


class ContinuousRangeSpec(StrictModel):
    """Continuous uniform range generator."""

    type: Literal["continuous_range"]
    start: float
    end: float

    @model_validator(mode="after")
    def validate_bounds(self) -> ContinuousRangeSpec:
        if self.end < self.start:
            raise ValueError("continuous_range requires end >= start")
        return self


class ContinuousLogRangeSpec(StrictModel):
    """Continuous log-uniform range generator."""

    type: Literal["continuous_log_range"]
    base: float
    start_exp: float
    end_exp: float

    @model_validator(mode="after")
    def validate_bounds(self) -> ContinuousLogRangeSpec:
        if self.base <= 0 or self.base == 1:
            raise ValueError("continuous_log_range requires base > 0 and base != 1")
        if self.end_exp < self.start_exp:
            raise ValueError("continuous_log_range requires end_exp >= start_exp")
        return self


SearchSpaceValue = (
    list[Any]
    | DiscreteRangeSpec
    | IntRangeSpec
    | LogRangeSpec
    | ContinuousRangeSpec
    | ContinuousLogRangeSpec
)


class DataConfig(StrictModel):
    """Input data locations and column names."""

    metadata_path: str = "testdata/c4_tiny/species_metadata.tsv"
    tpm_path: str = "testdata/c4_tiny/tpm.tsv"
    tree_path: str | None = None
    orthogroup_annotation_path: str | None = None
    species_col: str = "species"
    feature_col: str = "orthogroup"
    value_col: str = "tpm"
    trait_col: str = "C4"
    contrast_pair_col: str | None = "contrast_pair_id"


class SplitConfig(StrictModel):
    """Data split and CV controls."""

    group_col: str = "contrast_pair_id"
    exclude_col: str | None = None
    require_both_labels_per_group: bool = False
    outer_cv_strategy: OuterCvStrategy = "logo"
    outer_cv_n_splits: PositiveInt | None = None

    @model_validator(mode="after")
    def validate_group_kfold_args(self) -> SplitConfig:
        split_count_strategies = {"group_kfold", "stratified_group_kfold"}
        if self.outer_cv_strategy in split_count_strategies:
            if self.outer_cv_n_splits is None:
                raise ValueError(
                    "split.outer_cv_n_splits is required "
                    "when outer_cv_strategy=group_kfold|stratified_group_kfold"
                )
            if self.outer_cv_n_splits < 2:
                raise ValueError(
                    "split.outer_cv_n_splits must be >= 2 for "
                    "group_kfold|stratified_group_kfold"
                )
        elif self.outer_cv_n_splits is not None:
            raise ValueError(
                "split.outer_cv_n_splits is only valid "
                "when outer_cv_strategy=group_kfold|stratified_group_kfold"
            )
        return self


class SparseFeatureFilterConfig(StrictModel):
    """Sparse feature filter settings."""

    enabled: bool = True
    min_nonzero_fraction: float | None = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
    )
    scope: SparseFeatureScope = "any_trait"

    @model_validator(mode="after")
    def validate_enabled_args(self) -> SparseFeatureFilterConfig:
        if self.enabled and self.min_nonzero_fraction is None:
            raise ValueError(
                "preprocess.sparse_feature_filter."
                "min_nonzero_fraction "
                "is required when enabled=true"
            )
        return self


class LowVarianceFilterConfig(StrictModel):
    """Low variance feature filter settings."""

    enabled: bool = False
    min_variance: NonNegativeFloat | None = None

    @model_validator(mode="after")
    def validate_enabled_args(self) -> LowVarianceFilterConfig:
        if self.enabled and self.min_variance is None:
            raise ValueError(
                "preprocess.low_variance_filter.min_variance "
                "is required when enabled=true"
            )
        return self


class CorrelationFilterConfig(StrictModel):
    """Feature correlation filter settings."""

    enabled: bool = False
    method: CorrelationMethod = "pearson"
    max_abs_correlation: float | None = None

    @model_validator(mode="after")
    def validate_enabled_args(self) -> CorrelationFilterConfig:
        if self.enabled and self.max_abs_correlation is None:
            raise ValueError(
                "preprocess.correlation_filter.max_abs_correlation is required when enabled=true"
            )
        if self.max_abs_correlation is not None and not (0 < self.max_abs_correlation <= 1):
            raise ValueError("preprocess.correlation_filter.max_abs_correlation must be in (0, 1]")
        return self


class RankedFeatureFilterConfig(StrictModel):
    """Train-only ranked feature filter settings."""

    method: RankedFeatureFilterMethod = "none"
    max_features: PositiveInt | None = None
    min_contrast_pairs: PositiveInt = 1
    higher_in_trait: Literal[0, 1] | None = None

    @field_validator("higher_in_trait", mode="before")
    @classmethod
    def reject_boolean_higher_in_trait(cls, value: Any) -> Any:
        if isinstance(value, bool):
            raise ValueError(
                "preprocess.ranked_feature_filter.higher_in_trait must be 0, 1, or null"
            )
        return value

    @model_validator(mode="after")
    def validate_method_args(self) -> RankedFeatureFilterConfig:
        if self.method != "none" and self.max_features is None:
            raise ValueError(
                "preprocess.ranked_feature_filter.max_features is required "
                "when method is not none"
            )
        if (
            self.higher_in_trait is not None
            and self.method not in {"pair_aware", "unpaired"}
        ):
            raise ValueError(
                "preprocess.ranked_feature_filter.higher_in_trait is only configurable "
                "for method=pair_aware|unpaired"
            )
        return self


class ExpressionTransformConfig(StrictModel):
    """Sample x feature expression value transform settings."""

    method: ExpressionTransformMethod = "log1p"


class FeatureScalingConfig(StrictModel):
    """Train-fitted feature scaling settings."""

    method: FeatureScalingMethod = "standard"


class MissingExpressionConfig(StrictModel):
    """Optional observed-only standardization and neutral missing inputs."""

    method: Literal["none", "neutral"] = "none"
    zero_as_missing: bool = False


class AbstentionConfig(StrictModel):
    """Fixed information-coverage gate; no threshold fitting."""

    enabled: bool = False
    threshold: float = Field(default=0.8, gt=0.0, le=1.0, allow_inf_nan=False)

    @field_validator("threshold", mode="before")
    @classmethod
    def reject_boolean_threshold(cls, value: Any) -> Any:
        if isinstance(value, bool):
            raise ValueError("abstention.threshold must be a number in (0, 1]")
        return value


class PreprocessConfig(StrictModel):
    """Preprocessing settings."""

    max_pivot_cells: PositiveInt = 50_000_000
    absent_feature_fill: AbsentFeatureFill = 0
    missing_expression: MissingExpressionConfig = Field(default_factory=MissingExpressionConfig)
    expression_transform: ExpressionTransformConfig = Field(
        default_factory=ExpressionTransformConfig
    )
    sparse_feature_filter: SparseFeatureFilterConfig = Field(
        default_factory=SparseFeatureFilterConfig
    )
    low_variance_filter: LowVarianceFilterConfig = Field(default_factory=LowVarianceFilterConfig)
    ranked_feature_filter: RankedFeatureFilterConfig = Field(
        default_factory=RankedFeatureFilterConfig
    )
    correlation_filter: CorrelationFilterConfig = Field(default_factory=CorrelationFilterConfig)
    feature_scaling: FeatureScalingConfig = Field(default_factory=FeatureScalingConfig)

    @field_validator("absent_feature_fill", mode="before")
    @classmethod
    def reject_boolean_absent_feature_fill(cls, value: Any) -> Any:
        if isinstance(value, bool):
            raise ValueError("preprocess.absent_feature_fill must be 0 or nan")
        return value


class ModelConfig(StrictModel):
    """Model family selection."""

    name: ModelName = "logistic_elasticnet"


class SamplingConfig(StrictModel):
    """Sample-set construction and weighting policy."""

    strategy: SamplingStrategy = "group_balanced"
    max_samples_per_label_per_group: PositiveInt | None = 1
    sampled_set_count: PositiveInt = 10
    training_group_count: PositiveInt | None = None
    group_subsample_repeats: PositiveInt = 1
    group_subsample_repeat_index: PositiveInt = 1
    weighting: WeightingMode = "none"

    @model_validator(mode="after")
    def validate_sampling_compatibility(self) -> SamplingConfig:
        if self.strategy == "all_samples":
            if self.max_samples_per_label_per_group is not None:
                raise ValueError(
                    "sampling.max_samples_per_label_per_group must be null "
                    "when sampling.strategy=all_samples"
                )
            if self.sampled_set_count != 1:
                raise ValueError(
                    "sampling.sampled_set_count must be 1 "
                    "when sampling.strategy=all_samples"
                )
        return self


class EnsembleConfig(StrictModel):
    """Ensemble output controls."""

    probability_aggregation: ProbabilityAggregation = "mean"


class ModelSelectionConfig(StrictModel):
    """Hyperparameter candidate generation and selection settings."""

    selected_candidate_count: PositiveInt | None = None
    selected_candidate_percent: PositiveFloat | None = None
    candidate_source_policy: CandidateSourcePolicy = "per_sample_set"
    search_strategy: SearchStrategy = "grid"
    trial_count: PositiveInt | None = None
    search_space: dict[str, SearchSpaceValue] = Field(default_factory=dict)
    inner_cv_strategy: OuterCvStrategy | None = None
    inner_cv_n_splits: PositiveInt | None = None
    selection_metric: SelectionMetricName = "log_loss"
    selection_rule: SelectionRule = "best"

    @property
    def has_continuous_search_space(self) -> bool:
        """Whether search space contains at least one continuous parameter."""
        return any(
            isinstance(value, (ContinuousRangeSpec, ContinuousLogRangeSpec))
            for value in self.search_space.values()
        )

    @model_validator(mode="after")
    def validate_search_and_selection(self) -> ModelSelectionConfig:
        for param_name, param_value in self.search_space.items():
            if isinstance(param_value, list) and not param_value:
                raise ValueError(
                    f"model_selection.search_space.{param_name} cannot be an empty list"
                )

        if (
            self.selected_candidate_count is not None
            and self.selected_candidate_percent is not None
        ):
            raise ValueError(
                "model_selection.selected_candidate_count and "
                "model_selection.selected_candidate_percent are mutually exclusive"
            )
        if (
            self.selected_candidate_percent is not None
            and float(self.selected_candidate_percent) > 100.0
        ):
            raise ValueError("model_selection.selected_candidate_percent must be <= 100")

        if self.search_strategy in {"random", "tpe"} and self.trial_count is None:
            raise ValueError(
                "model_selection.trial_count is required when "
                "search_strategy=random|tpe"
            )

        if self.search_strategy == "grid" and self.has_continuous_search_space:
            raise ValueError(
                "model_selection.search_strategy=grid does not support "
                "continuous_range/continuous_log_range"
            )

        split_count_strategies = {"group_kfold", "stratified_group_kfold"}
        if self.inner_cv_strategy in split_count_strategies:
            if self.inner_cv_n_splits is None:
                raise ValueError(
                    "model_selection.inner_cv_n_splits is required "
                    "when inner_cv_strategy=group_kfold|stratified_group_kfold"
                )
            if self.inner_cv_n_splits < 2:
                raise ValueError(
                    "model_selection.inner_cv_n_splits must be >= 2 for "
                    "group_kfold|stratified_group_kfold"
                )
        elif self.inner_cv_n_splits is not None:
            raise ValueError(
                "model_selection.inner_cv_n_splits is only valid when "
                "inner_cv_strategy=group_kfold|stratified_group_kfold"
            )

        if (
            self.selected_candidate_count is not None
            or self.selected_candidate_percent is not None
        ) and self.inner_cv_strategy is None:
            raise ValueError(
                "model_selection.inner_cv_strategy is required when "
                "selected_candidate_count/selected_candidate_percent is set"
            )

        return self


class ReportConfig(StrictModel):
    """Prediction threshold settings."""

    pass


class GroupBootstrapConfig(StrictModel):
    """OOF group-bootstrap confidence interval controls."""

    enabled: bool = False
    n_resamples: PositiveInt = 2000
    confidence_level: float = Field(default=0.95, gt=0.0, lt=1.0)


class EvaluationConfig(StrictModel):
    """Evaluation uncertainty controls."""

    group_bootstrap: GroupBootstrapConfig = Field(default_factory=GroupBootstrapConfig)


class SummaryConfig(StrictModel):
    """Stage-level grouped summary controls."""

    group_col: str = Field(default="family", min_length=1)


class FiguresConfig(StrictModel):
    """Figure output controls."""

    top_features: PositiveInt = Field(default=30, le=100)


class RuntimeConfig(StrictModel):
    """Runtime execution controls."""

    seed: int = 42
    n_jobs: int = 1
    execution_stage: ExecutionStage = "cv_only"

    @model_validator(mode="after")
    def validate_runtime(self) -> RuntimeConfig:
        if self.n_jobs < 1:
            raise ValueError("runtime.n_jobs must be >= 1")
        return self


class PhylogeneticImputationConfig(StrictModel):
    """Optional interpretation of unknown-species predictions using observed traits."""

    enabled: bool = False
    branch_length_mode: Literal["input", "unit"] = "input"
    model: Literal["ER", "ARD"] = "ER"
    root_prior: Literal["equal", "empirical"] = "equal"


class AppConfig(StrictModel):
    """Top-level application configuration."""

    data: DataConfig = Field(default_factory=DataConfig)
    split: SplitConfig = Field(default_factory=SplitConfig)
    sampling: SamplingConfig = Field(default_factory=SamplingConfig)
    preprocess: PreprocessConfig = Field(default_factory=PreprocessConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    abstention: AbstentionConfig = Field(default_factory=AbstentionConfig)
    model_selection: ModelSelectionConfig = Field(default_factory=ModelSelectionConfig)
    ensemble: EnsembleConfig = Field(default_factory=EnsembleConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    summary: SummaryConfig = Field(default_factory=SummaryConfig)
    figures: FiguresConfig = Field(default_factory=FiguresConfig)
    phylogenetic_imputation: PhylogeneticImputationConfig = Field(
        default_factory=PhylogeneticImputationConfig
    )
    report: ReportConfig = Field(default_factory=ReportConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)

    @model_validator(mode="after")
    def validate_contrast_pair_dependencies(self) -> AppConfig:
        missing = self.preprocess.missing_expression
        if missing.method == "neutral":
            if (
                self.model.name != "logistic_elasticnet"
                or self.preprocess.expression_transform.method != "log1p"
                or self.preprocess.feature_scaling.method != "standard"
                or self.preprocess.absent_feature_fill != "nan"
            ):
                raise ValueError(
                    "preprocess.missing_expression.method=neutral requires "
                    "logistic_elasticnet, log1p, standard scaling, and absent_feature_fill=nan"
                )
        elif missing.zero_as_missing or self.abstention.enabled:
            raise ValueError(
                "zero_as_missing and abstention require "
                "preprocess.missing_expression.method=neutral"
            )
        if (
            self.preprocess.absent_feature_fill == "nan"
            and self.model.name != "random_forest"
            and missing.method != "neutral"
        ):
            raise ValueError(
                "preprocess.absent_feature_fill=nan is only supported when "
                "model.name=random_forest"
            )
        if (
            self.preprocess.ranked_feature_filter.method == "pair_aware"
            and self.data.contrast_pair_col is None
        ):
            raise ValueError(
                "preprocess.ranked_feature_filter.method=pair_aware requires "
                "data.contrast_pair_col"
            )
        selection_active = (
            self.model_selection.selected_candidate_count is not None
            or self.model_selection.selected_candidate_percent is not None
        )
        training_group_count = self.sampling.training_group_count
        if selection_active and training_group_count is not None:
            inner_strategy = self.model_selection.inner_cv_strategy
            required_groups = (
                2
                if inner_strategy == "logo"
                else self.model_selection.inner_cv_n_splits
            )
            if required_groups is not None and training_group_count < required_groups:
                raise ValueError(
                    "sampling.training_group_count must be at least the number of "
                    "groups required by model-selection inner CV; "
                    f"required={required_groups}"
                )
        if self.model.name == "logistic_elasticnet":
            supported = {"lambda", "alpha", "maxit", "thresh"}
            unsupported = sorted(set(self.model_selection.search_space) - supported)
            if unsupported:
                raise ValueError(
                    "Unsupported model_selection.search_space parameter(s) for "
                    f"logistic_elasticnet: {', '.join(unsupported)}. "
                    "Use glmnet parameters lambda, alpha, maxit, thresh; "
                    "lambda controls regularization directly (larger means stronger)."
                )
        return self
