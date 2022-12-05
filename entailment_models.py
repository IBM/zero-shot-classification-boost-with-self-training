# © Copyright IBM Corporation 2022.
#
# LICENSE: Apache License 2.0 (Apache-2.0)
# http://www.apache.org/licenses/LICENSE-2.0

import os
import random
import uuid

import torch

from datasets import Dataset
from tqdm.auto import tqdm
from transformers import (AutoModelForSequenceClassification, AutoTokenizer, PretrainedConfig, PreTrainedTokenizerBase,
                          InputFeatures, Trainer, TrainingArguments, RobertaConfig, BartConfig, DebertaConfig, pipeline)
from transformers.pipelines.pt_utils import KeyDataset


from data_structs import Predictions, SelfTrainingSet
from utils import get_root_dir


def get_zero_shot_predictions(model_name, texts_to_infer, label_names, batch_size, max_length):
    device = 0 if torch.cuda.is_available() else -1

    # We initialize the tokenizer here in order to set the maximum sequence length
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, model_max_length=max_length)

    classifier = pipeline("zero-shot-classification", model=model_name, tokenizer=tokenizer, device=device)

    ds = Dataset.from_dict({'text': texts_to_infer})

    preds_list = []
    for text, output in tqdm(zip(texts_to_infer, classifier(KeyDataset(ds, 'text'),
                                                            batch_size=batch_size,
                                                            candidate_labels=label_names, multi_label=True)),
                             total=len(ds), desc="zero-shot inference"):
        preds_list.append(output)

    predictions = Predictions(predicted_labels=[x['labels'][0] for x in preds_list],
                              ranked_classes=[x['labels'] for x in preds_list],
                              class_name_to_score=[dict(zip(x['labels'], x['scores'])) for x in preds_list])
    return predictions


def finetune_entailment_model(model_name, self_training_set: SelfTrainingSet, seed, learning_rate=2e-5, batch_size=32,
                              max_length=512, num_epochs=1, hypothesis_template="This example is {}."):
    model_id = f"{str(uuid.uuid4())}_fine_tuned_{model_name.replace(os.sep, '_')}"
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, model_max_length=max_length)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    out_dir = os.path.join(get_root_dir(), "output", "models", str(model_id))
    os.makedirs(out_dir, exist_ok=True)

    inputs = preprocess_and_tokenize(model.config, tokenizer, self_training_set, seed=seed,
                                     hypothesis_template=hypothesis_template)

    training_args = TrainingArguments(output_dir=out_dir,
                                      overwrite_output_dir=True,
                                      num_train_epochs=num_epochs,
                                      per_device_train_batch_size=batch_size,
                                      learning_rate=learning_rate)

    trainer = Trainer(model=model, args=training_args, train_dataset=inputs)
    trainer.train()

    trainer.save_model(out_dir)
    tokenizer.save_pretrained(out_dir)
    return out_dir


def preprocess_and_tokenize(model_config: PretrainedConfig, tokenizer: PreTrainedTokenizerBase,
                            self_training_set: SelfTrainingSet, seed: int, hypothesis_template: str):

    if type(model_config) not in [RobertaConfig, DebertaConfig, BartConfig]:
        raise NotImplementedError(f"{model_config.architectures} model is not supported")

    def get_numeric_label(label: str, model_config: PretrainedConfig):
        """
        Different entailment classification models on Huggingface use different names and IDs for the textual entailment
        labels of entailment/neutral/contradiction. Here we convert names to the appropriate model label IDs.
        """
        if label in model_config.label2id:
            return model_config.label2id[label]
        elif label.lower() in model_config.label2id:
            return model_config.label2id[label.lower()]
        else:
            raise Exception(f'The label "{label}" is not recognized by the model, '
                            f'model labels are: {model_config.label2id.keys()}')

    numeric_entailment_labels = [get_numeric_label(label, model_config)
                                 for label in self_training_set.entailment_labels]

    tokenized = []
    for text, class_name, label \
            in zip(self_training_set.texts, self_training_set.class_names, numeric_entailment_labels):

        hypothesis = hypothesis_template.format(class_name)
        inputs = (tokenizer.encode_plus([text, hypothesis], add_special_tokens=True, padding='max_length',
                                        truncation='only_first'))

        if type(model_config) == DebertaConfig:
            tokenized.append(InputFeatures(input_ids=inputs['input_ids'],
                                           attention_mask=inputs['attention_mask'],
                                           token_type_ids=inputs['token_type_ids'],
                                           label=label))
        elif type(model_config) in [RobertaConfig, BartConfig]:
            tokenized.append(InputFeatures(input_ids=inputs['input_ids'],
                                           attention_mask=inputs['attention_mask'],
                                           label=label))

    random.Random(seed).shuffle(tokenized)
    return tokenized
