import itertools
import logging
import time
from typing import List

import numpy as np
from tqdm import tqdm

from ca.datasets import Dataset, Task, load_corpus
from ca.evaluation import (calculate_binary_classification_score,
                           calculate_conll_2003_score)
from ca.experiments.simulation.simulation_util import (ModelType,
                                                       SimulationResult,
                                                       get_model,
                                                       save_simulation_results)
from ca.featurizer import CachedSentenceTransformer, CrfFeaturizer
from ca.strategies import StrategyType, get_strategy
from ca.util import DISABLE_TQDM, setup_logging

SEED_SIZE = 10

flatten = itertools.chain.from_iterable


def simulate_chunked(
    model_type: ModelType, dataset_name: Dataset, strategy_type: StrategyType, debug=False, debug_limit=30
) -> SimulationResult:
    corpus = load_corpus(dataset_name)

    if strategy_type == StrategyType.ANNOTATION_TIME and not corpus.has_time:
        raise ValueError("Corpus does not have time for strategy")

    model = get_model(model_type)
    strategy = get_strategy(strategy_type)

    df_train = corpus.train
    df_eval = corpus.test

    scores_eval = []
    t = []
    runtimes = []
    label_dict = {}

    if corpus.task == Task.SEQUENCE_TAGGING:
        logging.info("Task was sentence tagging")
        X_unannotated, y_unannotated = df_train.tag.group_by_documents_x_y()
        X_eval, y_eval = df_eval.tag.split_x_y_sentencewise()

        featurizer = CrfFeaturizer()
        Xf_unannotated = [featurizer.featurize(doc) for doc in X_unannotated]
        Xf_eval = featurizer.featurize(X_eval)

        label_dict = {
            lab: i for i, lab in enumerate(list(set([lab for doc in y_unannotated for sent in doc for lab in sent])))
        }

        # These need to be a list of sentences, but they are sentences grouped by document, therefore we flatten
        X_so_far = list(flatten(X_unannotated[:SEED_SIZE]))
        Xf_so_far = list(flatten(Xf_unannotated[:SEED_SIZE]))
        y_so_far = list(flatten(y_unannotated[:SEED_SIZE]))

        X_unannotated, Xf_unannotated, y_unannotated = (
            X_unannotated[SEED_SIZE:],
            Xf_unannotated[SEED_SIZE:],
            y_unannotated[SEED_SIZE:],
        )
    elif corpus.task == Task.DOCUMENT_CLASSIFICATION:
        logging.info("Task was document classification")
        X_unannotated, y_unannotated = df_train.dclass.split_x_y()
        X_eval, y_eval = df_eval.dclass.split_x_y()

        label_dict = {lab: i for i, lab in enumerate(list(set(y_unannotated)))}

        # TODO: Make configurable
        featurizer = CachedSentenceTransformer("bert-base-nli-mean-tokens")
        Xf_unannotated = featurizer.featurize(X_unannotated)
        Xf_eval = featurizer.featurize(X_eval)

        X_so_far, Xf_so_far, y_so_far = X_unannotated[:SEED_SIZE], Xf_unannotated[:SEED_SIZE], y_unannotated[:SEED_SIZE]
        X_unannotated, Xf_unannotated, y_unannotated = (
            X_unannotated[SEED_SIZE:],
            Xf_unannotated[SEED_SIZE:],
            y_unannotated[SEED_SIZE:],
        )
    elif corpus.task == Task.PAIRWISE_CLASSIFICATION:
        args1_train, args2_train, y_unannotated = df_train.pair.split_args_y()
        args1_eval, args2_eval, y_eval = df_eval.pair.split_args_y()

        X_unannotated = [arg1 + " [SEP] " + arg2 for arg1, arg2 in zip(args1_train, args2_train)]
        X_eval = [arg1 + " [SEP] " + arg2 for arg1, arg2 in zip(args1_eval, args2_eval)]

        featurizer = CachedSentenceTransformer("bert-base-nli-mean-tokens")

        Xf_unannotated = np.hstack([featurizer.featurize(args1_train), featurizer.featurize(args2_train)]).tolist()
        Xf_eval = np.hstack([featurizer.featurize(args1_eval), featurizer.featurize(args2_eval)]).tolist()

        X_so_far, Xf_so_far, y_so_far = X_unannotated[:SEED_SIZE], Xf_unannotated[:SEED_SIZE], y_unannotated[:SEED_SIZE]
        X_unannotated, Xf_unannotated, y_unannotated = (
            X_unannotated[SEED_SIZE:],
            Xf_unannotated[SEED_SIZE:],
            y_unannotated[SEED_SIZE:],
        )
    else:
        raise ValueError(f"Invalid task type: {corpus.task.name}")

    assert len(X_unannotated) == len(Xf_unannotated) == len(y_unannotated)
    assert len(X_eval) == len(Xf_eval) == len(y_eval)

    n = len(Xf_unannotated)

    # Get the annotation time information
    if corpus.has_time and strategy_type == StrategyType.ANNOTATION_TIME:
        logging.info("Using annotation time")

        if corpus.task == Task.SEQUENCE_TAGGING:
            times_so_far: List[int] = df_train.tag.get_times_per_document()[:SEED_SIZE]
            t_unannotated = df_train.tag.get_times_per_document()[SEED_SIZE:]
        elif corpus.task == Task.DOCUMENT_CLASSIFICATION or corpus.task == Task.PAIRWISE_CLASSIFICATION:
            times_so_far: List[int] = df_train[:SEED_SIZE].dclass.get_time_per_sentence()
            t_unannotated = df_train[SEED_SIZE:].dclass.get_time_per_sentence()
        else:
            raise ValueError(f"Invalid task type: {corpus.task.name}")

        assert (
            len(Xf_unannotated) == len(y_unannotated) == len(t_unannotated)
        ), f"{len(Xf_unannotated)} != {len(y_unannotated)} != {len(t_unannotated)}"

    # Setup strategy
    # Set label dict for Model difficulty:
    if strategy_type == StrategyType.MODEL_DIFFICULTY:
        logging.info("Labels for training {}".format(label_dict))
        strategy.set_label_dict(label_dict)

    # We pass references to lists, so strategies see the updated list after popping
    if corpus.task == Task.SEQUENCE_TAGGING:
        strategy.init_tagging(X_so_far, Xf_so_far, y_so_far, X_unannotated, Xf_unannotated, X_eval, Xf_eval)
    elif corpus.task == Task.DOCUMENT_CLASSIFICATION:
        strategy.init_document_classification(
            X_so_far, Xf_so_far, y_so_far, X_unannotated, Xf_unannotated, X_eval, Xf_eval
        )
    elif corpus.task == Task.PAIRWISE_CLASSIFICATION:
        strategy.init_pairwise(X_so_far, Xf_so_far, y_so_far, X_unannotated, Xf_unannotated, X_eval, Xf_eval)
    else:
        raise ValueError(f"Invalid task type: {corpus.task.name}")

    if strategy_type == StrategyType.ANNOTATION_TIME:
        strategy.set_time(times_so_far)

    # Model difficulty stuff:
    if strategy_type == StrategyType.MODEL_DIFFICULTY:
        strategy.init_model(100)
        strategy.init_training_params(epochs=5, batch_size=5)

    train_chunk_size = 10
    eval_chunk_size = 100
    for i in tqdm(range(0, n, train_chunk_size), disable=DISABLE_TQDM):
        indices = strategy.argsort_unchosen(train_chunk_size)
        assert len(indices) == len(X_unannotated) == len(Xf_unannotated) == len(y_unannotated)

        # 1. Select the next nth chunk to annotate
        # 2. Select the next nth chunk to annotate
        # 3. Use the model trained on chunks 0 .. n-1 to evaluate on chunk n
        # 4. Add the nth chunk to the training data

        next_train_chunk_indices = indices[:train_chunk_size]
        next_eval_chunk_indices = indices[:eval_chunk_size]

        X_train_chunk = [X_unannotated[idx] for idx in next_train_chunk_indices]
        Xf_train_chunk = [Xf_unannotated[idx] for idx in next_train_chunk_indices]
        y_train_chunk = [y_unannotated[idx] for idx in next_train_chunk_indices]

        X_eval_chunk = [X_unannotated[idx] for idx in next_eval_chunk_indices]
        Xf_eval_chunk = [Xf_unannotated[idx] for idx in next_eval_chunk_indices]
        y_eval_chunk = [y_unannotated[idx] for idx in next_eval_chunk_indices]

        # Train and evaluate
        begin = time.process_time()
        model.fit(Xf_so_far, y_so_far)
        end = time.process_time()

        if corpus.task == Task.SEQUENCE_TAGGING:
            y_eval_pred = model.predict(list(flatten(Xf_eval_chunk)))
        else:
            y_eval_pred = model.predict(Xf_eval_chunk)

        # Compute scores
        if corpus.task == Task.DOCUMENT_CLASSIFICATION or corpus.task == Task.PAIRWISE_CLASSIFICATION:
            score = calculate_binary_classification_score(y_eval_chunk, y_eval_pred)
            # logging.info(f"P: {score.precision} R: {score.recall} F1: {score.f1}")
        elif corpus.task == Task.SEQUENCE_TAGGING:
            # TODO: WNUT uses different score than conll2003
            score = calculate_conll_2003_score(list(flatten(y_eval_chunk)), y_eval_pred)
        else:
            raise ValueError(f"Invalid task type: {corpus.task.name}")

        scores_eval.append(score)
        t.append(i)
        runtimes.append(end - begin)

        # Update datasets
        if corpus.task == Task.SEQUENCE_TAGGING:
            X_so_far.extend(flatten(list(X_train_chunk)))
            Xf_so_far.extend(flatten(list(Xf_train_chunk)))
            y_so_far.extend(flatten(list(y_train_chunk)))
        else:
            X_so_far.extend(X_train_chunk)
            Xf_so_far.extend(Xf_train_chunk)
            y_so_far.extend(y_train_chunk)

        X_unannotated[:] = X_unannotated[train_chunk_size:]
        Xf_unannotated[:] = Xf_unannotated[train_chunk_size:]
        y_unannotated[:] = y_unannotated[train_chunk_size:]

        if strategy_type == StrategyType.ANNOTATION_TIME:
            t_chunk = [t_unannotated[idx] for idx in next_train_chunk_indices]
            t_unannotated[:] = t_unannotated[train_chunk_size:]
            times_so_far.extend(t_chunk)
            assert len(X_so_far) == len(Xf_so_far) == len(y_so_far) == len(times_so_far)
            assert len(X_unannotated) == len(Xf_unannotated) == len(y_unannotated) == len(t_unannotated)

        assert len(X_unannotated) == len(Xf_unannotated) == len(y_unannotated)

        if debug and i >= debug_limit:
            break

    return SimulationResult(corpus.name, model_type, strategy_type, t, scores_eval, runtimes)


def main():
    setup_logging()

    DEBUG = True
    DEBUG_LIMIT = 50

    experiments_muc7t = [
        (Dataset.MUC7T, ModelType.SKLEARN_CRF_TAGGER, StrategyType.RANDOM),
        (Dataset.MUC7T, ModelType.SKLEARN_CRF_TAGGER, StrategyType.ANNOTATION_TIME),
        (Dataset.MUC7T, ModelType.SKLEARN_CRF_TAGGER, StrategyType.FLESCH_KINCAID_READING_EASE),
        (Dataset.MUC7T, ModelType.SKLEARN_CRF_TAGGER, StrategyType.FLESCH_KINCAID_GRADE_LEVEL),
    ]

    experiments_sigie = [
        (Dataset.SIGIE, ModelType.SKLEARN_CRF_TAGGER, StrategyType.ANNOTATION_TIME),
        (Dataset.SIGIE, ModelType.SKLEARN_CRF_TAGGER, StrategyType.FLESCH_KINCAID_GRADE_LEVEL),
        (Dataset.SIGIE, ModelType.SKLEARN_CRF_TAGGER, StrategyType.RANDOM),
    ]

    experiments_spec = [
        (Dataset.SPEC, ModelType.LIGHTGBM_CLASSIFIER, StrategyType.ANNOTATION_TIME),
        (Dataset.SPEC, ModelType.LIGHTGBM_CLASSIFIER, StrategyType.RANDOM),
        (Dataset.SPEC, ModelType.LIGHTGBM_CLASSIFIER, StrategyType.FLESCH_KINCAID_GRADE_LEVEL),
    ]

    experiments_convincing = [
        (Dataset.CONVINCING, ModelType.LIGHTGBM_CLASSIFIER, StrategyType.RANDOM),
        (Dataset.CONVINCING, ModelType.LIGHTGBM_CLASSIFIER, StrategyType.ANNOTATION_TIME),
        (Dataset.CONVINCING, ModelType.LIGHTGBM_CLASSIFIER, StrategyType.FLESCH_KINCAID_GRADE_LEVEL),
    ]

    # Dataset, Model, Strategy
    experiments = [
        # (Dataset.FAMULUS_EDA_MED, ModelType.SKLEARN_CRF_TAGGER, StrategyType.BERT_PREDICTION_DIFFICULTY),
        # (Dataset.FAMULUS_EDA_MED, ModelType.SKLEARN_CRF_TAGGER, StrategyType.MODEL_DIFFICULTY),
    ]

    experiments = experiments_muc7t + experiments_convincing + experiments_spec + experiments_sigie

    results = []

    for dataset_name, model_type, strategy in experiments:
        logging.info("Simulating [%s] using [%s] with [%s]", dataset_name, model_type, strategy)
        result = simulate_chunked(model_type, dataset_name, strategy, DEBUG, DEBUG_LIMIT)
        results.append(result)

    save_simulation_results(results)


if __name__ == "__main__":
    main()
