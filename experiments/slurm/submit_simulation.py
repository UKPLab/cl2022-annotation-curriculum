import submitit

from ca.datasets import Dataset
from ca.experiments.simulation.run_chunked_simulation import simulate_chunked
from ca.experiments.simulation.run_simulation import simulate
from ca.experiments.simulation.simulation_util import (ModelType,
                                                       save_simulation_results)
from ca.strategies import StrategyType
from ca.util import setup_logging

setup_logging()

log_folder = "slurm_logs/%j"
executor = submitit.AutoExecutor(folder=log_folder)

timeout = 60*24 # 1 day
executor.update_parameters(slurm_partition="ukp", mem_gb=16, cpus_per_task=4, name="curriculum_annotation", timeout_min=timeout, slurm_timeout_min=timeout)

experiments_muc7t = [(Dataset.MUC7T, ModelType.SKLEARN_CRF_TAGGER, StrategyType.RANDOM)] * 5 + \
                    [(Dataset.MUC7T, ModelType.SKLEARN_CRF_TAGGER, StrategyType.ANNOTATION_TIME)] * 5 + \
                    [(Dataset.MUC7T, ModelType.SKLEARN_CRF_TAGGER, StrategyType.FLESCH_KINCAID_READING_EASE)]

experiments_sigie = [(Dataset.SIGIE, ModelType.SKLEARN_CRF_TAGGER, StrategyType.RANDOM)] * 5 + \
                    [(Dataset.SIGIE, ModelType.SKLEARN_CRF_TAGGER, StrategyType.ANNOTATION_TIME)] * 5 + \
                    [(Dataset.SIGIE, ModelType.SKLEARN_CRF_TAGGER, StrategyType.FLESCH_KINCAID_READING_EASE)]

experiments_spec = [(Dataset.SPEC, ModelType.LIGHTGBM_CLASSIFIER, StrategyType.RANDOM)] * 5 + \
                   [(Dataset.SPEC, ModelType.LIGHTGBM_CLASSIFIER, StrategyType.ANNOTATION_TIME)] * 5 + \
                   [(Dataset.SPEC, ModelType.LIGHTGBM_CLASSIFIER, StrategyType.FLESCH_KINCAID_READING_EASE)]

experiments = experiments_spec + experiments_sigie + experiments_muc7t

DEBUG = False
DEBUG_LIMIT = 10

jobs = []
with executor.batch():
    for dataset_name, model_type, strategy in experiments:
        job1 = executor.submit(simulate, model_type, dataset_name, strategy, DEBUG, DEBUG_LIMIT)
        #job2 = executor.submit(simulate_chunked, model_type, dataset_name, strategy, DEBUG, DEBUG_LIMIT)

        jobs.append(job1)
        #jobs.append(job2)

results = [job.result() for job in jobs]

save_simulation_results(results)
