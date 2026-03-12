"""Pydantic schema for configuration validation."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeFloat,
    PositiveInt,
    model_validator,
)

ExecutionStage = Literal["cv_only", "full_run"]
OuterCvStrategy = Literal["logo", "group_kfold"]
SamplingStrategy = Literal["all_samples", "group_balanced"]
WeightingMode = Literal["none", "group_label_inverse"]
ModelName = Literal["logistic_elasticnet", "linear_svm", "random_forest"]
ProbabilityAggregation = Literal["mean", "median"]
SearchStrategy = Literal["grid", "random", "tpe"]
CandidateSourcePolicy = Literal["per_sample_set", "reuse_first_sample_set"]
SelectionMetricName = Literal["mcc", "balanced_accuracy", "log_loss"]
ThresholdSelectionMetricName = Literal["mcc", "balanced_accuracy"]
CorrelationMethod = Literal["pearson", "spearman"]


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
    species_col: str = "species"
    feature_col: str = "orthogroup"
    value_col: str = "tpm"
    trait_col: str = "C4"
    group_col: str = "contrast_pair_id"


class SplitConfig(StrictModel):
    """Data split and CV controls."""

    outer_cv_strategy: OuterCvStrategy = "logo"
    outer_cv_n_splits: PositiveInt | None = None

    @model_validator(mode="after")
    def validate_group_kfold_args(self) -> SplitConfig:
        if self.outer_cv_strategy == "group_kfold":
            if self.outer_cv_n_splits is None:
                raise ValueError(
                    "split.outer_cv_n_splits is required "
                    "when outer_cv_strategy=group_kfold"
                )
            if self.outer_cv_n_splits < 2:
                raise ValueError("split.outer_cv_n_splits must be >= 2 for group_kfold")
        elif self.outer_cv_n_splits is not None:
            raise ValueError(
                "split.outer_cv_n_splits is only valid "
                "when outer_cv_strategy=group_kfold"
            )
        return self


class LowPrevalenceFilterConfig(StrictModel):
    """Low prevalence feature filter settings."""

    enabled: bool = True
    min_species_per_feature: PositiveInt | None = 2

    @model_validator(mode="after")
    def validate_enabled_args(self) -> LowPrevalenceFilterConfig:
        if self.enabled and self.min_species_per_feature is None:
            raise ValueError(
                "preprocess.low_prevalence_filter.min_species_per_feature "
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


class PreprocessConfig(StrictModel):
    """Preprocessing settings."""

    max_pivot_cells: PositiveInt = 50_000_000
    low_prevalence_filter: LowPrevalenceFilterConfig = Field(
        default_factory=LowPrevalenceFilterConfig
    )
    low_variance_filter: LowVarianceFilterConfig = Field(default_factory=LowVarianceFilterConfig)
    correlation_filter: CorrelationFilterConfig = Field(default_factory=CorrelationFilterConfig)


class ModelConfig(StrictModel):
    """Model family selection."""

    name: ModelName = "logistic_elasticnet"


class SamplingConfig(StrictModel):
    """Sample-set construction and weighting policy."""

    strategy: SamplingStrategy = "group_balanced"
    max_samples_per_label_per_group: PositiveInt | None = 1
    sampled_set_count: PositiveInt = 10
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
    candidate_source_policy: CandidateSourcePolicy = "reuse_first_sample_set"
    search_strategy: SearchStrategy = "grid"
    trial_count: PositiveInt | None = None
    search_space: dict[str, SearchSpaceValue] = Field(default_factory=dict)
    inner_cv_strategy: OuterCvStrategy | None = None
    inner_cv_n_splits: PositiveInt | None = None
    selection_metric: SelectionMetricName = "log_loss"

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

        if self.inner_cv_strategy == "group_kfold":
            if self.inner_cv_n_splits is None:
                raise ValueError(
                    "model_selection.inner_cv_n_splits is required "
                    "when inner_cv_strategy=group_kfold"
                )
            if self.inner_cv_n_splits < 2:
                raise ValueError("model_selection.inner_cv_n_splits must be >= 2 for group_kfold")
        elif self.inner_cv_n_splits is not None:
            raise ValueError(
                "model_selection.inner_cv_n_splits is only valid when inner_cv_strategy=group_kfold"
            )

        if self.selected_candidate_count is not None and self.inner_cv_strategy is None:
            raise ValueError(
                "model_selection.inner_cv_strategy is required when selected_candidate_count is set"
            )

        return self


class ReportConfig(StrictModel):
    """Threshold derivation settings."""

    fixed_probability_threshold: float = 0.5
    auto_threshold_selection_metric: ThresholdSelectionMetricName = "mcc"

    @model_validator(mode="after")
    def validate_thresholds(self) -> ReportConfig:
        if not (0 <= self.fixed_probability_threshold <= 1):
            raise ValueError("report.fixed_probability_threshold must be between 0 and 1")
        return self


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


class AppConfig(StrictModel):
    """Top-level application configuration."""

    data: DataConfig = Field(default_factory=DataConfig)
    split: SplitConfig = Field(default_factory=SplitConfig)
    sampling: SamplingConfig = Field(default_factory=SamplingConfig)
    preprocess: PreprocessConfig = Field(default_factory=PreprocessConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    model_selection: ModelSelectionConfig = Field(default_factory=ModelSelectionConfig)
    ensemble: EnsembleConfig = Field(default_factory=EnsembleConfig)
    report: ReportConfig = Field(default_factory=ReportConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
