![license](https://img.shields.io/github/license/IBM/zero-shot-classification-boost-with-self-training)  ![python](https://img.shields.io/badge/python-3.9-blue)
# Zero-shot Classification Boost with Self-training

Code to reproduce the zero-shot self-training experiments from [Gera et al. (2022)](#reference). 

Using this repository you can: 

1. Download the datasets used in the paper;
2. Run NLI-based zero-shot text classifiers;
3. Fine-tune an NLI model using a set of pseudo-labeled examples in an iterative self-training procedure;
4. Compare the classification performance between the base model and the self-training iterations.


**Table of contents**

[Installation](#installation)

[Running an experiment](#running-an-experiment)

[Reference](#reference)

[License](#license)

## Installation
The framework requires Python 3.9
1. Clone the repository locally: 
   `git clone https://github.com/IBM/zero-shot-classification-boost-with-self-training.git`
2. Go to the cloned directory 
  `cd zero-shot-classification-boost-with-self-training`
3. Install the project dependencies: `pip install -r requirements.txt`
4. Run the python script `python download_data.py`.
This script downloads and processes the classification datasets used in the paper.

                         
## Running an experiment
The experiment script `run_self_training_experiment.py` requires 3 arguments:
- `experiment_name`: an identifier for the experiment
- `dataset_name`: the name of one of the datasets under the "datasets/" directory
- `base_model`: the name of an NLI-based zero-shot classifier from the [Hugging Face hub](https://huggingface.co/models?pipeline_tag=zero-shot-classification)

In addition, there are several optional configuration parameters:
- `num_iterations`: the number of self-training iterations to perform. Defaults to 2 (as used in the paper).
- `dataset_subset_size`: an upper limit to the size of the unlabeled set used for self-training, in order to reduce runtime. Defaults to 10000 (as used in the paper).
- `sample_ratio`: the desired proportion from the unlabeled set to collect as pseudo-labeled entailment examples for each target class. Defaults to 0.01 (as used in the paper; with a `dataset_subset_size` of 10000 this would mean up to 100 positive entailment examples per class).
- `negative_sampling_strategy`: the strategy for adding negative (contradiction) pseudo-labeled examples. Defaults to "take_random". The other options are "take_all", "take_second", and "take_last". See the paper for more details.
- `train_batch_size`: for the fine-tuning of the NLI models. Defaults to 16.
Note that the maximal possible batch size would depend on the amount of GPU memory available and the base model used. In our experimental setup, using NVIDIA A100-80GB GPUs, we ran with a batch size of 32 for the RoBERTa and BART NLI models, and 16 for the DeBERTa model.
- `infer_batch_size`: for the NLI model evaluation. Defaults to 16.
- `max_length`: the maximum sequence length of examples sent to the NLI model, used for both inference and fine-tuning. Defaults to 512. Note that in the paper experiments we (unintentionally) used a sequence length of 512 for inference and 256 for the fine-tuning examples. As in the batch size parameter, the maximal possible sequence length would depend on the amount of GPU memory available and the base model used.
- `seed`: the random seed used for model fine-tuning and for random sampling operations. For robust experimental results we recommend running several repetitions of each dataset/self-training configuration using different seeds.
- `delete_models`: whether to delete fine-tuned models after training and evaluation is complete (in order to save disk space). False by default.

For example: 

```python run_self_training_experiment.py --experiment_name my_experiment --dataset_name yahoo_answers --base_model roberta-large-mnli --seed 0```

For each experimental run and for each self-training iteration, the zero-shot classification accuracy over the `test.csv` file of the dataset is written to the screen as well as to `output/experiments/<experiment_name>/all_copies.csv`. 

Multiple runs can safely write in parallel to the same `all_copies.csv` file - each new result is appended to the file. In addition, for every new result, an aggregation of all the results so far is written to `output/experiments/<experiment_name>/aggregated.csv`. This aggregation reflects the mean of all runs for each experimental setting (i.e. self-training iteration i of base model M over test dataset D, with a given set of configuration parameters).

NOTE: Following this work, we performed a thorough evaluation regarding the robustness and contribution of the masking module described in the paper. We found the contribution due to masking to be small in relation to the rest of the self-training framework. Thus, it was not included in this implementation to keep it as lean and practical as possible. In case anyone is specifically interested in reproducing the masking results, feel free to contact the authors or open an issue.

## Running on additional datasets
The repository enables downloading the paper datasets with `python download_data.py`. In order to run an experiment on a new dataset, create a folder under `datasets/` with the following files:
- `class_names.txt`: a list of all the dataset target class names, separated by newlines.
- `unlabeled.csv`: a csv with a `text` column, containing texts to be used for creating the pseudo-labeled self-training set.
- `test.csv`: a csv with labeled data for model evaluation, which contains a `text` column, as well as a `label` column specifying the class name of the correct target class for the text.

## Reference
Ariel Gera, Alon Halfon, Eyal Shnarch, Yotam Perlitz, Liat Ein-Dor and Noam Slonim (2022). 
[Zero-Shot Text Classification with Self-Training](https://arxiv.org/abs/2210.17541). EMNLP 2022

Please cite: 
```
@inproceedings{gera2022zero,
  title={Zero-Shot Text Classification with Self-Training},
  author={Gera, Ariel and Halfon, Alon and Shnarch, Eyal and Perlitz, Yotam and Ein-Dor, Liat and Slonim, Noam},
  booktitle={Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing},
  publisher={Association for Computational Linguistics},
  url={https://arxiv.org/abs/2210.17541},
  year={2022}
}
```

## License
This work is released under the Apache 2.0 license. The full text of the license can be found in [LICENSE](LICENSE).
