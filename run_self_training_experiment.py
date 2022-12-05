# © Copyright IBM Corporation 2022.
#
# LICENSE: Apache License 2.0 (Apache-2.0)
# http://www.apache.org/licenses/LICENSE-2.0

import logging
import os
import random
import shutil

from argparse import ArgumentParser
from collections import defaultdict, Counter
from enum import Enum
from typing import Dict, List

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s')

import pandas as pd

from data_structs import SelfTrainingSet, Predictions
from entailment_models import finetune_entailment_model, get_zero_shot_predictions
from utils import set_seed, get_root_dir
from evaluate import evaluate_classification_performance


class NegativeSamplingStrategy(Enum):
    TAKE_ALL = 0
    TAKE_RANDOM = 1
    TAKE_SECOND = 2
    TAKE_LAST = 3


def rank_candidate_indices_per_class(all_class_names, predictions: Predictions) -> Dict[str, List[int]]:
    diff_scores_to_second_best = \
        [class_name_to_score[ranked_classes[0]] - class_name_to_score[ranked_classes[1]]
         for class_name_to_score, ranked_classes in zip(predictions.class_name_to_score, predictions.ranked_classes)]

    class_name_to_sorted_candidate_idxs = {}
    for class_name in all_class_names:
        sorted_candidate_idxs = \
            [idx for idx, (predicted_class, diff_to_second_best)
             in sorted(enumerate(zip(predictions.predicted_labels, diff_scores_to_second_best)),
                       key=lambda x: x[1][1], reverse=True)
             if predicted_class == class_name]
        class_name_to_sorted_candidate_idxs[class_name] = sorted_candidate_idxs

    return class_name_to_sorted_candidate_idxs


def get_negative_examples(predictions: Predictions, class_name_to_chosen_pos_idxs: Dict[str, List[int]],
                          negative_sampling_strategy: NegativeSamplingStrategy) -> Dict[str, List[int]]:

    all_positive_idxs = [idx for class_idxs in class_name_to_chosen_pos_idxs.values() for idx in class_idxs]

    class_name_to_chosen_negative_idxs = defaultdict(list)
    for idx in all_positive_idxs:
        example_ranked_classes = predictions.ranked_classes[idx]

        if negative_sampling_strategy == NegativeSamplingStrategy.TAKE_SECOND:
            class_name_to_chosen_negative_idxs[example_ranked_classes[1]].append(idx)
        elif negative_sampling_strategy == NegativeSamplingStrategy.TAKE_ALL:
            for class_name in example_ranked_classes[1:]:
                class_name_to_chosen_negative_idxs[class_name].append(idx)
        elif negative_sampling_strategy == NegativeSamplingStrategy.TAKE_LAST:
            class_name_to_chosen_negative_idxs[example_ranked_classes[-1]].append(idx)
        elif negative_sampling_strategy == NegativeSamplingStrategy.TAKE_RANDOM:
            random_negative_class = random.choice(example_ranked_classes[1:])
            class_name_to_chosen_negative_idxs[random_negative_class].append(idx)
        else:
            raise ValueError(f"Unknown negative sampling strategy {negative_sampling_strategy}")

    return class_name_to_chosen_negative_idxs


if __name__ == '__main__':
    # TODO missing datasets (amazon, goemotions)
    # TODO decide whether to include masking
    # TODO plot?

    parser = ArgumentParser()
    parser.add_argument('--experiment_name', required=True)
    parser.add_argument('--dataset_name', required=True)
    parser.add_argument("--base_model", required=True)

    parser.add_argument("--num_iterations", type=int, default=2)
    parser.add_argument("--dataset_subset_size", type=int, default=10000)
    parser.add_argument("--sample_ratio", type=float, default=0.01)
    parser.add_argument("--negative_sampling_strategy", default='take_random', type=str)

    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--train_batch_size", type=int, default=16)
    parser.add_argument("--infer_batch_size", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument('--seed', type=int, default=0)

    parser.add_argument("--delete_models", action='store_true')

    args = parser.parse_args()
    config_dict = vars(args)
    logging.info(config_dict)

    # This string describes the full self-training configuration, to ease aggregation across seeds
    setting_name = '_'.join([str(value) for key, value in config_dict.items()
                             if key not in ['seed', 'infer_batch_size', 'delete_models']])
    set_seed(args.seed)

    data_path = os.path.join(get_root_dir(), 'datasets', args.dataset_name)
    out_dir = os.path.join(get_root_dir(), 'output', 'experiments', args.experiment_name)
    os.makedirs(out_dir, exist_ok=True)

    unlabeled_df = pd.read_csv(os.path.join(data_path, 'unlabeled.csv'))
    unlabeled_texts = unlabeled_df['text']

    with open(os.path.join(data_path, 'class_names.txt')) as f:
        class_names = f.read().splitlines()

    # Limit the size of the unlabeled set to reduce runtime
    subset_idxs = random.sample(range(len(unlabeled_texts)), min(args.dataset_subset_size, len(unlabeled_texts)))
    unlabeled_texts = [unlabeled_texts[idx] for idx in subset_idxs]

    test_df = pd.read_csv(os.path.join(data_path, 'test.csv'))
    test_texts = test_df['text']
    test_gold_labels = test_df['label']

    # Set the desired number of pseudo-labeled positive examples per class
    sample_size = int(len(unlabeled_texts) * args.sample_ratio)
    logging.info(f'sample size per class is {sample_size}, set by a sample ratio of {args.sample_ratio}')

    model_name = args.base_model

    logging.info(f"Evaluating base zero-shot model '{model_name}' performance on test set")
    test_preds = get_zero_shot_predictions(model_name, test_texts, class_names,
                                           batch_size=args.infer_batch_size, max_length=args.max_length)
    evaluate_classification_performance(test_preds.predicted_labels, test_gold_labels, out_dir,
                                        info_dict={
                                            'iteration': 0,
                                            'setting_name': f'{setting_name}_base',
                                            **config_dict
                                        })

    for iter_number in range(1, args.num_iterations+1):
        logging.info(f"Inferring with zero-shot model '{model_name}' on {len(unlabeled_texts)} unlabeled elements)")
        predictions = get_zero_shot_predictions(model_name, unlabeled_texts, class_names,
                                                batch_size=args.infer_batch_size, max_length=args.max_length)
        logging.info(f"Done inferring zero-shot model on {len(unlabeled_texts)} unlabeled elements")

        if args.delete_models and model_name != args.base_model:
            logging.info(f"deleting fine-tuned model {model_name}")
            shutil.rmtree(model_name)

        self_training_set = SelfTrainingSet()
        # For each class, we rank the elements as candidates for self-training according to the model confidence
        class_name_to_sorted_idxs = rank_candidate_indices_per_class(class_names, predictions)

        # We choose the <sample_size> best examples from each class as positive (entailment) examples
        class_name_to_positive_chosen_idxs = {class_name: sorted_idxs[:sample_size]
                                              for class_name, sorted_idxs in class_name_to_sorted_idxs.items()}

        for class_name, idxs in class_name_to_positive_chosen_idxs.items():
            self_training_set.texts.extend([unlabeled_texts[idx] for idx in idxs])
            self_training_set.class_names.extend([class_name]*len(idxs))
            self_training_set.entailment_labels.extend(['ENTAILMENT']*len(idxs))

        # Add negative (contradiction) examples
        negative_sampling_strategy = NegativeSamplingStrategy[args.negative_sampling_strategy.upper()]
        class_name_to_negative_chosen_idxs = \
            get_negative_examples(predictions, class_name_to_positive_chosen_idxs, negative_sampling_strategy)

        for class_name, idxs in class_name_to_negative_chosen_idxs.items():
            self_training_set.texts.extend([unlabeled_texts[idx] for idx in idxs])
            self_training_set.class_names.extend([class_name]*len(idxs))
            self_training_set.entailment_labels.extend(['CONTRADICTION']*len(idxs))

        logging.info(f"Done collecting pseudo-labeled elements for self-training iteration {iter_number}: "
                     f"{Counter(self_training_set.entailment_labels)}")

        # We use the updated pseudo-labeled set from this iteration to fine-tune the *base* entailment model
        logging.info(f"Fine-tuning model '{args.base_model}' on {len(self_training_set.entailment_labels)} "
                     f"pseudo-labeled texts")
        finetuned_model_path = finetune_entailment_model(
            model_name=args.base_model, self_training_set=self_training_set, seed=args.seed,
            learning_rate=args.learning_rate, batch_size=args.train_batch_size, max_length=args.max_length,
            num_epochs=1)
        logging.info(f"Done fine-tuning. Model for self-training iteration {iter_number} "
                     f"saved to {finetuned_model_path}.")

        model_name = finetuned_model_path

        logging.info(f'iteration {iter_number}: evaluating model {model_name} performance on test set')
        test_preds = get_zero_shot_predictions(model_name, test_texts, class_names,
                                               batch_size=args.infer_batch_size, max_length=args.max_length)
        evaluate_classification_performance(test_preds.predicted_labels, test_gold_labels, out_dir,
                                            info_dict={
                                                'iteration': iter_number,
                                                'setting_name': f'{setting_name}_iter_{iter_number}',
                                                'self_training_set_size':  len(self_training_set.texts),
                                                **config_dict
                                            })
