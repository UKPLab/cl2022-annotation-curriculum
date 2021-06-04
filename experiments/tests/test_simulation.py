import pytest

from ca.datasets import Dataset
from ca.experiments.simulation.run_simulation import simulate
from ca.experiments.simulation.simulation_util import ModelType
from ca.strategies import StrategyType


@pytest.mark.parametrize("dataset_name, model_type, strategy", [
    (Dataset.CONVINCING, ModelType.LIGHTGBM_CLASSIFIER, StrategyType.RANDOM),
    (Dataset.CONVINCING, ModelType.LIGHTGBM_CLASSIFIER, StrategyType.FLESCH_KINCAID_GRADE_LEVEL),
    (Dataset.CONVINCING, ModelType.LIGHTGBM_CLASSIFIER, StrategyType.ANNOTATION_TIME),

    (Dataset.SIGIE, ModelType.SKLEARN_CRF_TAGGER, StrategyType.ANNOTATION_TIME),
    (Dataset.SIGIE, ModelType.SKLEARN_CRF_TAGGER, StrategyType.FLESCH_KINCAID_GRADE_LEVEL),
    (Dataset.SIGIE, ModelType.SKLEARN_CRF_TAGGER, StrategyType.RANDOM),

    (Dataset.SPEC, ModelType.LIGHTGBM_CLASSIFIER, StrategyType.ANNOTATION_TIME),
    (Dataset.SPEC, ModelType.LIGHTGBM_CLASSIFIER, StrategyType.RANDOM),
    (Dataset.SPEC, ModelType.LIGHTGBM_CLASSIFIER, StrategyType.FLESCH_KINCAID_GRADE_LEVEL),
])
def test_simulation_smoketest(dataset_name: Dataset, model_type: ModelType, strategy: StrategyType):
    simulate(model_type, dataset_name, strategy, debug=True, debug_limit=30)