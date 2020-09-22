# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

import os, logging
import mxnet as mx

def get_movielens_data(data_dir, prefix):
    # MovieLens 10M dataset from https://grouplens.org/datasets/movielens/
    # This dataset is copy right to GroupLens Research Group at the University of Minnesota,
    # and licensed under their usage license.
    # For full text of the license, see http://files.grouplens.org/datasets/movielens/ml-10m-README.html
    if not os.path.exists(os.path.join(data_dir, "ml-10M100K")):
        mx.test_utils.get_zip_data(data_dir,
                                   "http://files.grouplens.org/datasets/movielens/%s.zip" % prefix,
                                   prefix + ".zip")
        assert os.path.exists(os.path.join(data_dir, "ml-10M100K"))
        os.system("cd data/ml-10M100K; chmod +x allbut.pl; sh split_ratings.sh; cd -;")

def get_movielens_iter(filename, batch_size):
    """Not particularly fast code to parse the text file and load into NDArrays.
    return two data iters, one for train, the other for validation.
    """
    logging.info("Preparing data iterators for " + filename + " ... ")
    user = []
    item = []
    score = []
    with open(filename, 'r') as f:
        num_samples = 0
        for line in f:
            tks = line.strip().split('::')
            if len(tks) != 4:
                continue
            num_samples += 1
            user.append((tks[0]))
            item.append((tks[1]))
            score.append((tks[2]))
    # convert to ndarrays
    user = mx.nd.array(user, dtype='int32')
    item = mx.nd.array(item)
    score = mx.nd.array(score)
    # prepare data iters
    data_train = {'user': user, 'item': item}
    label_train = {'score': score}
    iter_train = mx.io.NDArrayIter(data=data_train,label=label_train,
                                   batch_size=batch_size, shuffle=True)
    return mx.io.PrefetchingIter(iter_train)


