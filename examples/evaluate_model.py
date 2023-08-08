# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

"""
This example shows how to fit a model and evaluate its predictions.
"""
import pprint

from gluonts.dataset.repository import get_dataset, dataset_recipes
from gluonts.evaluation import make_evaluation_predictions, Evaluator
from gluonts.mx import SimpleFeedForwardEstimator, DeepAREstimator
from gluonts.mx.trainer import Trainer
from gluonts.mx.distribution import DistributionOutput, StudentTOutput, GaussianOutput



if __name__ == "__main__":
    print(f"datasets available: {dataset_recipes.keys()}")

    # we pick m4_hourly as it only contains a few hundred time series
    # dataset = get_dataset("m4_hourly", regenerate=False)
    dataset = get_dataset("constant", regenerate=False)

    # estimator = SimpleFeedForwardEstimator(
    #     prediction_length=dataset.metadata.prediction_length,
    #     trainer=Trainer(epochs=5, num_batches_per_epoch=10),
    # )

    estimator = DeepAREstimator(
        prediction_length=dataset.metadata.prediction_length,
        trainer=Trainer(epochs=5, num_batches_per_epoch=10, hybridize=False),
        freq=dataset.metadata.freq,
        distr_output=GaussianOutput(),
        # distr_output=StudentTOutput(),
    )

    predictor = estimator.train(dataset.train)

    forecast_it, ts_it = make_evaluation_predictions(
        dataset.test, predictor=predictor, num_samples=100
    )

    agg_metrics, item_metrics = Evaluator()(
        ts_it, forecast_it, num_series=len(dataset.test)
    )

    pprint.pprint(agg_metrics)
