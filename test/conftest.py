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


from pathlib import Path
from typing import List, NamedTuple
import gluonts
import logging
import numpy as np
import os
import pandas as pd
import pytest
import random
import sys
import warnings

from gluonts.dataset.common import ListDataset

try:
    import mxnet as mx
except ImportError:
    mx = None


class HierarchicalMetaData(NamedTuple):
    S: np.ndarray
    freq: str
    nodes: List


class HierarchicalTrainDatasets(NamedTuple):
    train: ListDataset
    test: ListDataset
    metadata: HierarchicalMetaData


@pytest.fixture
def sine7(seq_length: int = 100, prediction_length: int = 10):
    x = np.arange(0, seq_length)

    # Bottom layer (4 series)
    amps = [0.8, 0.9, 1, 1.1]
    freqs = [1 / 20, 1 / 30, 1 / 50, 1 / 100]

    b = np.zeros((4, seq_length))
    for i, f in enumerate(freqs):
        omega = 0
        if i == 3:
            np.random.seed(0)
            omega = np.random.uniform(0, np.pi)  # random phase shift
        b[i, :] = amps[i] * np.sin(2 * np.pi * x * f + omega)

    # Aggregation matrix S
    S = np.array(
        [
            [1, 1, 1, 1],
            [1, 1, 0, 0],
            [0, 0, 1, 1],
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ],
    )

    Y = S @ b

    # Indices and timestamps
    index = pd.date_range(
        start=pd.Timestamp("2020-01-01", freq="D"),
        periods=Y.shape[1],
        freq="D",
    )

    metadata = HierarchicalMetaData(
        S=S, freq=index.freqstr, nodes=[2, [2] * 2]
    )

    train_dataset = ListDataset(
        [
            {
                "start": index[0],
                "item_id": "all_items",
                "target": Y[:, :-prediction_length],
            }
        ],
        freq=index.freqstr,
        one_dim_target=False,
    )

    test_dataset = ListDataset(
        [{"start": index[0], "item_id": "all_items", "target": Y}],
        freq=index.freqstr,
        one_dim_target=False,
    )

    assert Y.shape[0] == S.shape[0]
    return HierarchicalTrainDatasets(
        train=train_dataset, test=test_dataset, metadata=metadata
    )


@pytest.fixture(scope="function", autouse=True)
def function_scope_seed(request):
    """A function scope fixture that manages rng seeds.
    This fixture automatically initializes the python, numpy and mxnet random
    number generators randomly on every test run.
    def test_ok_with_random_data():
        ...
    To fix the seed used for a test case mark the test function with the
    desired seed:
    @pytest.mark.seed(1)
    def test_not_ok_with_random_data():
        '''This testcase actually works.'''
        assert 17 == random.randint(0, 100)
    When a test fails, the fixture outputs the seed used. The user can then set
    the environment variable MXNET_TEST_SEED to the value reported, then rerun
    the test with:
        pytest --verbose -s <test_module_name.py> -k <failing_test>
    To run a test repeatedly, install pytest-repeat and add the --count argument:
        pip install pytest-repeat
        pytest --verbose -s <test_module_name.py> -k <failing_test> --count 1000
    """

    seed = request.node.get_closest_marker("seed")
    env_seed_str = os.getenv("MXNET_TEST_SEED")

    if seed is not None:
        seed = seed.args[0]
        assert isinstance(seed, int)
    elif env_seed_str is not None:
        seed = int(env_seed_str)
    else:
        seed = int.from_bytes(os.urandom(4), "big")

    post_test_state = np.random.get_state()
    np.random.seed(seed)
    random.seed(seed)

    if mx is not None:
        mx.random.seed(seed)

    seed_message = (
        "np/mx/python random seeds are set to "
        f"{seed}, use MXNET_TEST_SEED={seed} to reproduce."
    )

    # Always log seed on DEBUG log level. This makes sure we can find out the
    # value of the seed even if the test case causes a segfault and subsequent
    # teardown code is not run.
    logging.debug(seed_message)

    yield  # run the test

    np.random.set_state(post_test_state)


@pytest.fixture(autouse=True)
def doctest(doctest_namespace):
    doctest_namespace["np"] = np
    doctest_namespace["gluonts"] = gluonts

    if mx is not None:
        doctest_namespace["mx"] = mx
        doctest_namespace["gluon"] = mx.gluon

    import doctest

    doctest.ELLIPSIS_MARKER = "-etc-"


def get_collect_ignores():
    test_folder = Path(__file__).parent.resolve()

    old_path = sys.path
    sys.path = [
        path for path in sys.path if Path(path).resolve() != test_folder
    ]

    excludes = []

    for path in test_folder.glob("**/require-packages.txt"):
        with path.open() as requirements:
            for requirement in map(str.strip, requirements):
                try:
                    __import__(requirement)
                except ImportError:
                    excludes.append(str(path.parent.relative_to(test_folder)))
                    break

    if excludes:
        warnings.warn(
            f"Skipping tests because some packages are not installed: {excludes}"
        )

    sys.path = old_path
    return excludes


collect_ignore = get_collect_ignores()
