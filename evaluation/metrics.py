## Copied from https://github.com/tensorflow/models/blob/master/research/delf/delf/python/google_landmarks_dataset/metrics.py


# Copyright 2019 The TensorFlow Authors All Rights Reserved.
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
# ==============================================================================
"""Python module to compute metrics for Google Landmarks dataset."""

import numpy as np


def _count_positives(solution):
    """Counts number of test images with non-empty ground-truth in `solution`.
      Args:
        solution: Dict mapping test image ID to list of ground-truth IDs.
      Returns:
        count: Number of test images with non-empty ground-truth.
  """
    count = 0
    for v in solution.values():
        if v:
            count += 1

    return count


def get_solution_dict(test_id, db_id):
    """Computes the solution from query and database labels
      Args:
        test_id: List of test image IDs.
        db_id: List of database image IDs.
      Returns:
        solution: Dict mapping test image ID to list of
          ground-truth database image IDs.
    """
    solutions = {}
    for i, t_id in enumerate(test_id):
        solution = [str(j) for j, d_id in enumerate(db_id) if
                    (d_id == t_id) & (i != j)]
        solutions[str(i)] = solution
    return solutions


def get_prediction_dict(prediction_id):
    """Computes the solution from query and database labels
    Args:
      test_id: List of test image IDs.
      prediction_id: List of database image IDs.
    Returns:
      predictions: Dict mapping test image ID to a list of strings corresponding
        to index image IDs.
  """
    predictions = {}
    for i, p_id in enumerate(prediction_id):
        prediction = [str(p) for p in p_id if i!=p]
        predictions[str(i)] = prediction
    return predictions


def mean_average_precision(predictions, retrieval_solution, max_predictions=100):
    """Computes mean average precision for retrieval prediction.
      Args:
        predictions: Dict mapping test image ID to a list of strings corresponding
          to index image IDs.
        retrieval_solution: Dict mapping test image ID to list of ground-truth image
          IDs.
        max_predictions: Maximum number of predictions per query to take into
          account. For the Google Landmark Retrieval challenge, this should be set
          to 100.
      Returns:
        mean_ap: Mean average precision score (float).
      Raises:
        ValueError: If a test image in `predictions` is not included in
          `retrieval_solutions`.
  """
    # Compute number of test images.
    num_test_images = len(retrieval_solution.keys())

    # Loop over predictions for each query and compute mAP.
    mean_ap = 0.0
    for key, prediction in predictions.items():
        if key not in retrieval_solution:
            raise ValueError('Test image %s is not part of retrieval_solution' % key)

        # Loop over predicted images, keeping track of those which were already
        # used (duplicates are skipped).
        ap = 0.0
        already_predicted = set()
        num_expected_retrieved = min(len(retrieval_solution[key]), max_predictions)
        num_correct = 0
        for i in range(min(len(prediction), max_predictions)):
            if prediction[i] not in already_predicted:
                if prediction[i] in retrieval_solution[key]:
                    num_correct += 1
                    ap += num_correct / (i + 1)
                already_predicted.add(prediction[i])

        ap /= num_expected_retrieved
        mean_ap += ap

    mean_ap /= num_test_images

    return mean_ap


def mean_precisions(predictions, retrieval_solution, max_predictions=100):
    """Computes mean precisions for retrieval prediction.
      Args:
        predictions: Dict mapping test image ID to a list of strings corresponding
          to index image IDs.
        retrieval_solution: Dict mapping test image ID to list of ground-truth image
          IDs.
        max_predictions: Maximum number of predictions per query to take into
          account.
      Returns:
        mean_precisions: NumPy array with mean precisions at ranks 1 through
          `max_predictions`.
      Raises:
        ValueError: If a test image in `predictions` is not included in
          `retrieval_solutions`.
  """
    # Compute number of test images.
    num_test_images = len(retrieval_solution.keys())

    # Loop over predictions for each query and compute precisions@k.
    precisions = np.zeros((num_test_images, max_predictions))
    count_test_images = 0
    for key, prediction in predictions.items():
        if key not in retrieval_solution:
            raise ValueError('Test image %s is not part of retrieval_solution' % key)

        # Loop over predicted images, keeping track of those which were already
        # used (duplicates are skipped).
        already_predicted = set()
        num_correct = 0
        for i in range(max_predictions):
            if i < len(prediction):
                if prediction[i] not in already_predicted:
                    if prediction[i] in retrieval_solution[key]:
                        num_correct += 1
                    already_predicted.add(prediction[i])
            precisions[count_test_images, i] = num_correct / (i + 1)
        count_test_images += 1

    mean_precisions = np.mean(precisions, axis=0)

    return mean_precisions


def mean_median_position(predictions, retrieval_solution, max_predictions=100):
    """Computes mean and median positions of first correct image.
      Args:
        predictions: Dict mapping test image ID to a list of strings corresponding
          to index image IDs.
        retrieval_solution: Dict mapping test image ID to list of ground-truth image
          IDs.
        max_predictions: Maximum number of predictions per query to take into
          account.
      Returns:
        mean_position: Float.
        median_position: Float.
      Raises:
        ValueError: If a test image in `predictions` is not included in
          `retrieval_solutions`.
  """
    # Compute number of test images.
    num_test_images = len(retrieval_solution.keys())

    # Loop over predictions for each query to find first correct ranked image.
    positions = (max_predictions + 1) * np.ones((num_test_images))
    count_test_images = 0
    for key, prediction in predictions.items():
        if key not in retrieval_solution:
            raise ValueError('Test image %s is not part of retrieval_solution' % key)

        for i in range(min(len(prediction), max_predictions)):
            if prediction[i] in retrieval_solution[key]:
                positions[count_test_images] = i + 1
                break

        count_test_images += 1

    mean_position = np.mean(positions)
    median_position = np.median(positions)

    return mean_position, median_position


def recall_rate_at_k(results, query_labels, db_labels, k):
    """Evaluate Recall-Rate@k based on retrieval results
      Args:
          results:        numpy array of size [NUM_QUERY_IMAGES x k], indices of k nearest neighbors for each query
          query_labels:   list of labels for each query
          db_labels:      list of labels for each db
          k:              number of nn results to evaluate
      Returns:
          rr_at_k:    Recall-Rate@k in percentage
  """
    self_retrieval = False
    if query_labels is db_labels:
        self_retrieval = True
    expected_result_size = k + 1 if self_retrieval else k

    assert results.shape[1] >= expected_result_size, \
        "Not enough retrieved results to evaluate Recall@{}".format(k)
    rr_at_k = np.zeros((k,))
    for i in range(len(query_labels)):
        pos = 0  # keep track recall at pos
        j = 0  # looping through results
        while pos < k:
            if self_retrieval and i == results[i, j]:
                # Only skip the document when query and index sets are the exact same
                j += 1
                continue
            if query_labels[i] == db_labels[results[i, j]]:
                rr_at_k[pos:] += 1
                break
            j += 1
            pos += 1
    return rr_at_k / float(len(query_labels)) * 100.0
