# © Copyright IBM Corporation 2022.
#
# LICENSE: Apache License 2.0 (Apache-2.0)
# http://www.apache.org/licenses/LICENSE-2.0

import io
import logging
import os

import numpy as np
import pandas as pd
from filelock import FileLock
from sklearn.metrics import classification_report


def safe_to_csv(df, path, save_index_col=False):
    buffer = io.StringIO()
    df.to_csv(buffer, index=save_index_col)
    buffer.seek(0)
    output = buffer.getvalue()
    buffer.close()
    with open(path, "w") as text_file:
        text_file.write(output)
        text_file.flush()


def save_results(res_file_name, agg_file_name, result_dict, setting_name_col='setting_name'):
    def mean_str(col: pd.Series):
        if pd.api.types.is_numeric_dtype(col):
            return col.mean()
        else:
            return col.dropna().unique()[0] if col.nunique() == 1 else np.NaN

    new_res_df = pd.DataFrame([result_dict])

    lock_path = os.path.abspath(os.path.join(res_file_name, os.pardir, 'result_csv_files.lock'))
    with FileLock(lock_path):
        if os.path.isfile(res_file_name):
            orig_df = pd.read_csv(res_file_name)
            df = pd.concat([orig_df, new_res_df])
            df_agg = df.groupby(by=setting_name_col).agg(mean_str).sort_values(by=['dataset_name', setting_name_col])
            safe_to_csv(df_agg, agg_file_name, save_index_col=True)
        else:
            df = new_res_df
        safe_to_csv(df, res_file_name)


def evaluate_classification_performance(predicted_labels, gold_labels, out_dir, info_dict):

    accuracy = np.mean([gold_label == prediction for gold_label, prediction in zip(gold_labels, predicted_labels)])
    evaluation_dict = {'accuracy': accuracy, 'evaluation_size': len(gold_labels)}

    report = classification_report(gold_labels, predicted_labels, output_dict=True)
    report.pop('accuracy')
    for category, metrics in report.items():
        if not category.endswith("avg"):
            category = f"'{category}'"

        evaluation_dict[f"{category} precision"] = metrics['precision']
        evaluation_dict[f"{category} recall"] = metrics['recall']
        evaluation_dict[f"{category} f1"] = metrics['f1-score']

    evaluation_dict = {**info_dict, **evaluation_dict}
    logging.info(evaluation_dict)

    all_copies_file = os.path.join(out_dir, "all_copies.csv")
    agg_file = os.path.join(out_dir, "aggregated.csv")
    save_results(all_copies_file, agg_file, evaluation_dict)
