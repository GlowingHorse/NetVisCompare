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
from typing import Tuple

import torch
from torch import Tensor

from torchmetrics.utilities.checks import _input_format_classification
from torchmetrics.utilities.enums import DataType
from torchmetrics.utilities.imports import _TORCH_GREATER_EQUAL_1_8


def _binning_with_loop(
    confidences: Tensor, accuracies: Tensor, bin_boundaries: Tensor
) -> Tuple[Tensor, Tensor, Tensor]:
    """Compute calibration bins using for loops. Use for pytorch < 1.6.

    Args:
        confidences: The confidence (i.e. predicted prob) of the top1 prediction.
        accuracies: 1.0 if the top-1 prediction was correct, 0.0 otherwise.
        bin_boundaries: Bin boundaries separating the ``linspace`` from 0 to 1.

    Returns:
        tuple with binned accuracy, binned confidence and binned probabilities
    """
    conf_bin = torch.zeros_like(bin_boundaries)
    acc_bin = torch.zeros_like(bin_boundaries)
    prop_bin = torch.zeros_like(bin_boundaries)
    for i, (bin_lower, bin_upper) in enumerate(zip(bin_boundaries[:-1], bin_boundaries[1:])):
        # Calculated confidence and accuracy in each bin
        in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
        prop_in_bin = in_bin.float().mean()
        if prop_in_bin.item() > 0:
            acc_bin[i] = accuracies[in_bin].float().mean()
            conf_bin[i] = confidences[in_bin].mean()
            prop_bin[i] = prop_in_bin
    return acc_bin, conf_bin, prop_bin


def _binning_bucketize(
    confidences: Tensor, accuracies: Tensor, bin_boundaries: Tensor
) -> Tuple[Tensor, Tensor, Tensor]:
    """Compute calibration bins using ``torch.bucketize``. Use for pytorch >= 1.6.

    Args:
        confidences: The confidence (i.e. predicted prob) of the top1 prediction.
        accuracies: 1.0 if the top-1 prediction was correct, 0.0 otherwise.
        bin_boundaries: Bin boundaries separating the ``linspace`` from 0 to 1.

    Returns:
        tuple with binned accuracy, binned confidence and binned probabilities
    """
    acc_bin = torch.zeros(len(bin_boundaries) - 1, device=confidences.device, dtype=confidences.dtype)
    conf_bin = torch.zeros(len(bin_boundaries) - 1, device=confidences.device, dtype=confidences.dtype)
    count_bin = torch.zeros(len(bin_boundaries) - 1, device=confidences.device, dtype=confidences.dtype)

    indices = torch.bucketize(confidences, bin_boundaries) - 1

    count_bin.scatter_add_(dim=0, index=indices, src=torch.ones_like(confidences))

    conf_bin.scatter_add_(dim=0, index=indices, src=confidences)
    conf_bin = torch.nan_to_num(conf_bin / count_bin)

    acc_bin.scatter_add_(dim=0, index=indices, src=accuracies)
    acc_bin = torch.nan_to_num(acc_bin / count_bin)

    prop_bin = count_bin / count_bin.sum()
    return acc_bin, conf_bin, prop_bin


def _ce_compute(
    confidences: Tensor,
    accuracies: Tensor,
    bin_boundaries: Tensor,
    norm: str = "l1",
    debias: bool = False,
) -> Tensor:
    """Computes the calibration error given the provided bin boundaries and norm.

    Args:
        confidences: The confidence (i.e. predicted prob) of the top1 prediction.
        accuracies: 1.0 if the top-1 prediction was correct, 0.0 otherwise.
        bin_boundaries: Bin boundaries separating the ``linspace`` from 0 to 1.
        norm: Norm function to use when computing calibration error. Defaults to "l1".
        debias: Apply debiasing to L2 norm computation as in
            `Verified Uncertainty Calibration`_. Defaults to False.

    Raises:
        ValueError: If an unsupported norm function is provided.

    Returns:
        Tensor: Calibration error scalar.
    """
    if norm not in {"l1", "l2", "max"}:
        raise ValueError(f"Norm {norm} is not supported. Please select from l1, l2, or max. ")

    if _TORCH_GREATER_EQUAL_1_8:
        acc_bin, conf_bin, prop_bin = _binning_bucketize(confidences, accuracies, bin_boundaries)
    else:
        acc_bin, conf_bin, prop_bin = _binning_with_loop(confidences, accuracies, bin_boundaries)

    if norm == "l1":
        ce = torch.sum(torch.abs(acc_bin - conf_bin) * prop_bin)
    elif norm == "max":
        ce = torch.max(torch.abs(acc_bin - conf_bin))
    elif norm == "l2":
        ce = torch.sum(torch.pow(acc_bin - conf_bin, 2) * prop_bin)
        # NOTE: debiasing is disabled in the wrapper functions. This implementation differs from that in sklearn.
        if debias:
            # the order here (acc_bin - 1 ) vs (1 - acc_bin) is flipped from
            # the equation in Verified Uncertainty Prediction (Kumar et al 2019)/
            debias_bins = (acc_bin * (acc_bin - 1) * prop_bin) / (prop_bin * accuracies.size()[0] - 1)
            ce += torch.sum(torch.nan_to_num(debias_bins))  # replace nans with zeros if nothing appeared in a bin
        ce = torch.sqrt(ce) if ce > 0 else torch.tensor(0)
    return ce


def _ce_update(preds: Tensor, target: Tensor) -> Tuple[Tensor, Tensor]:
    """Given a predictions and targets tensor, computes the confidences of the top-1 prediction and records their
    correctness.

    Args:
        preds:  Input ``softmaxed`` predictions.
        target: Labels.

    Raises:
        ValueError: If the dataset shape is not binary, multiclass, or multidimensional-multiclass.

    Returns:
        tuple with confidences and accuracies
    """
    _, _, mode = _input_format_classification(preds, target)

    if mode == DataType.BINARY:
        if not ((0 <= preds) * (preds <= 1)).all():
            preds = preds.sigmoid()
        confidences, accuracies = preds, target
    elif mode == DataType.MULTICLASS:
        if not ((0 <= preds) * (preds <= 1)).all():
            preds = preds.softmax(dim=1)
        confidences, predictions = preds.max(dim=1)
        accuracies = predictions.eq(target)
    elif mode == DataType.MULTIDIM_MULTICLASS:
        # reshape tensors
        # for preds, move the class dimension to the final axis and flatten the rest
        confidences, predictions = torch.transpose(preds, 1, -1).flatten(0, -2).max(dim=1)
        # for targets, just flatten the target
        accuracies = predictions.eq(target.flatten())
    else:
        raise ValueError(
            f"Calibration error is not well-defined for data with size {preds.size()} and targets {target.size()}."
        )
    # must be cast to float for ddp allgather to work
    return confidences.float(), accuracies.float()


def calibration_error(preds: Tensor, target: Tensor, n_bins: int = 15, norm: str = "l1") -> Tensor:
    r"""`Computes the Top-label Calibration Error`_

    Three different norms are implemented, each corresponding to variations on the calibration error metric.

    L1 norm (Expected Calibration Error)

    .. math::
        \text{ECE} = \sum_i^N b_i \|(p_i - c_i)\|

    Infinity norm (Maximum Calibration Error)

    .. math::
        \text{MCE} =  \max_{i} (p_i - c_i)

    L2 norm (Root Mean Square Calibration Error)

    .. math::
        \text{RMSCE} = \sqrt{\sum_i^N b_i(p_i - c_i)^2}

    Where :math:`p_i` is the top-1 prediction accuracy in bin :math:`i`,
    :math:`c_i` is the average confidence of predictions in bin :math:`i`, and
    :math:`b_i` is the fraction of data points in bin :math:`i`.

    .. note:
        L2-norm debiasing is not yet supported.

    Args:
        preds: Model output probabilities.
        target: Ground-truth target class labels.
        n_bins: Number of bins to use when computing t.
        norm: Norm used to compare empirical and expected probability bins.
            Defaults to "l1", or Expected Calibration Error.
    """
    if norm not in ("l1", "l2", "max"):
        raise ValueError(f"Norm {norm} is not supported. Please select from l1, l2, or max. ")

    if not isinstance(n_bins, int) or n_bins <= 0:
        raise ValueError(f"Expected argument `n_bins` to be a int larger than 0 but got {n_bins}")

    confidences, accuracies = _ce_update(preds, target)

    bin_boundaries = torch.linspace(0, 1, n_bins + 1, dtype=torch.float, device=preds.device)

    return _ce_compute(confidences, accuracies, bin_boundaries, norm=norm)
