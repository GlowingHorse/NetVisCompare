# Copyright The PyTorch Lightning team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Any, List, Optional

import torch
from torch import Tensor

from torchmetrics.functional.classification.auroc import _auroc_compute, _auroc_update
from torchmetrics.metric import Metric
from torchmetrics.utilities import rank_zero_warn
from torchmetrics.utilities.data import dim_zero_cat
from torchmetrics.utilities.enums import DataType
from torchmetrics.utilities.imports import _TORCH_LOWER_1_6


class AUROC(Metric):
    r"""Compute Area Under the Receiver Operating Characteristic Curve (`ROC AUC`_).
    Works for both binary, multilabel and multiclass problems. In the case of
    multiclass, the values will be calculated based on a one-vs-the-rest approach.

    Forward accepts

    - ``preds`` (float tensor): ``(N, ...)`` (binary) or ``(N, C, ...)`` (multiclass) tensor
      with probabilities, where C is the number of classes.

    - ``target`` (long tensor): ``(N, ...)`` or ``(N, C, ...)`` with integer labels

    For non-binary input, if the ``preds`` and ``target`` tensor have the same
    size the input will be interpretated as multilabel and if ``preds`` have one
    dimension more than the ``target`` tensor the input will be interpretated as
    multiclass.

    .. note::
        If either the positive class or negative class is completly missing in the target tensor,
        the auroc score is meaningless in this case and a score of 0 will be returned together
        with an warning.

    Args:
        num_classes: integer with number of classes for multi-label and multiclass problems.

            Should be set to ``None`` for binary problems
        pos_label: integer determining the positive class. Default is ``None``
            which for binary problem is translated to 1. For multiclass problems
            this argument should not be set as we iteratively change it in the
            range ``[0, num_classes-1]``
        average:
            - ``'micro'`` computes metric globally. Only works for multilabel problems
            - ``'macro'`` computes metric for each class and uniformly averages them
            - ``'weighted'`` computes metric for each class and does a weighted-average,
              where each class is weighted by their support (accounts for class imbalance)
            - ``None`` computes and returns the metric per class
        max_fpr:
            If not ``None``, calculates standardized partial AUC over the
            range ``[0, max_fpr]``. Should be a float between 0 and 1.

        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Raises:
        ValueError:
            If ``average`` is none of ``None``, ``"macro"`` or ``"weighted"``.
        ValueError:
            If ``max_fpr`` is not a ``float`` in the range ``(0, 1]``.
        RuntimeError:
            If ``PyTorch version`` is ``below 1.6`` since ``max_fpr`` requires ``torch.bucketize``
            which is not available below 1.6.
        ValueError:
            If the mode of data (binary, multi-label, multi-class) changes between batches.

    Example (binary case):
        >>> from torchmetrics import AUROC
        >>> preds = torch.tensor([0.13, 0.26, 0.08, 0.19, 0.34])
        >>> target = torch.tensor([0, 0, 1, 1, 1])
        >>> auroc = AUROC(pos_label=1)
        >>> auroc(preds, target)
        tensor(0.5000)

    Example (multiclass case):
        >>> preds = torch.tensor([[0.90, 0.05, 0.05],
        ...                       [0.05, 0.90, 0.05],
        ...                       [0.05, 0.05, 0.90],
        ...                       [0.85, 0.05, 0.10],
        ...                       [0.10, 0.10, 0.80]])
        >>> target = torch.tensor([0, 1, 1, 2, 2])
        >>> auroc = AUROC(num_classes=3)
        >>> auroc(preds, target)
        tensor(0.7778)

    """
    is_differentiable: bool = False
    higher_is_better: bool = True
    full_state_update: bool = False
    preds: List[Tensor]
    target: List[Tensor]

    def __init__(
        self,
        num_classes: Optional[int] = None,
        pos_label: Optional[int] = None,
        average: Optional[str] = "macro",
        max_fpr: Optional[float] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        self.num_classes = num_classes
        self.pos_label = pos_label
        self.average = average
        self.max_fpr = max_fpr

        allowed_average = (None, "macro", "weighted", "micro")
        if self.average not in allowed_average:
            raise ValueError(
                f"Argument `average` expected to be one of the following: {allowed_average} but got {average}"
            )

        if self.max_fpr is not None:
            if not isinstance(max_fpr, float) or not 0 < max_fpr <= 1:
                raise ValueError(f"`max_fpr` should be a float in range (0, 1], got: {max_fpr}")

            if _TORCH_LOWER_1_6:
                raise RuntimeError(
                    "`max_fpr` argument requires `torch.bucketize` which is not available below PyTorch version 1.6"
                )

        self.mode: DataType = None  # type: ignore
        self.add_state("preds", default=[], dist_reduce_fx="cat")
        self.add_state("target", default=[], dist_reduce_fx="cat")

        rank_zero_warn(
            "Metric `AUROC` will save all targets and predictions in buffer."
            " For large datasets this may lead to large memory footprint."
        )

    def update(self, preds: Tensor, target: Tensor) -> None:  # type: ignore
        """Update state with predictions and targets.

        Args:
            preds: Predictions from model (probabilities, or labels)
            target: Ground truth labels
        """
        preds, target, mode = _auroc_update(preds, target)

        self.preds.append(preds)
        self.target.append(target)

        if self.mode and self.mode != mode:
            raise ValueError(
                "The mode of data (binary, multi-label, multi-class) should be constant, but changed"
                f" between batches from {self.mode} to {mode}"
            )
        self.mode = mode

    def compute(self) -> Tensor:
        """Computes AUROC based on inputs passed in to ``update`` previously."""
        if not self.mode:
            raise RuntimeError("You have to have determined mode.")
        preds = dim_zero_cat(self.preds)
        target = dim_zero_cat(self.target)
        return _auroc_compute(
            preds,
            target,
            self.mode,
            self.num_classes,
            self.pos_label,
            self.average,
            self.max_fpr,
        )
