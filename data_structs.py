# © Copyright IBM Corporation 2022.
#
# LICENSE: Apache License 2.0 (Apache-2.0)
# http://www.apache.org/licenses/LICENSE-2.0

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class Predictions:
    predicted_labels: List[str]
    ranked_classes: List[List[str]]
    class_name_to_score: List[Dict[str, float]]


@dataclass
class SelfTrainingSet:
    texts: List[str] = field(default_factory=list)
    class_names: List[str] = field(default_factory=list)
    entailment_labels: List[str] = field(default_factory=list)



