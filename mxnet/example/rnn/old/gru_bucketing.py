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

# pylint: disable=C0111,too-many-arguments,too-many-instance-attributes,too-many-locals,redefined-outer-name,fixme
# pylint: disable=superfluous-parens, no-member, invalid-name
import sys
sys.path.insert(0, "../../python")
import numpy as np
import mxnet as mx

from gru import gru_unroll
from bucket_io import BucketSentenceIter, default_build_vocab, DummyIter

def Perplexity(label, pred):
    label = label.T.reshape((-1,))
    loss = 0.
    for i in range(pred.shape[0]):
        loss += -np.log(max(1e-10, pred[i][int(label[i])]))
    return np.exp(loss / label.size)

if __name__ == '__main__':
    batch_size = 32
    #buckets = [10, 20, 30, 40, 50, 60]
    #buckets = [32]
    buckets = []
    num_hidden = 200
    num_embed = 200
    num_lstm_layer = 2

    num_epoch = 25
    learning_rate = 0.01
    momentum = 0.0

    # dummy data is used to test speed without IO
    dummy_data = False

    #contexts = [mx.context.gpu(i) for i in range(1)]
    contexts = mx.context.cpu()

    vocab = default_build_vocab("./data/sherlockholmes.train.txt")

    def sym_gen(seq_len):
        return gru_unroll(num_lstm_layer, seq_len, len(vocab),
                           num_hidden=num_hidden, num_embed=num_embed,
                           num_label=len(vocab))

    init_h = [('l%d_init_h'%l, (batch_size, num_hidden)) for l in range(num_lstm_layer)]

    data_train = BucketSentenceIter("./data/sherlockholmes.train.txt", vocab,
                                    buckets, batch_size, init_h)
    data_val = BucketSentenceIter("./data/sherlockholmes.valid.txt", vocab,
                                  buckets, batch_size, init_h)

    if dummy_data:
        data_train = DummyIter(data_train)
        data_val = DummyIter(data_val)

    if len(buckets) == 1:
        # only 1 bucket, disable bucketing
        symbol = sym_gen(buckets[0])
    else:
        symbol = sym_gen

    model = mx.model.FeedForward(ctx=contexts,
                                 symbol=symbol,
                                 num_epoch=num_epoch,
                                 learning_rate=learning_rate,
                                 momentum=momentum,
                                 wd=0.00001,
                                 initializer=mx.init.Xavier(factor_type="in", magnitude=2.34))

    import logging
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=head)

    model.fit(X=data_train, eval_data=data_val,
              eval_metric = mx.metric.np(Perplexity),
              batch_end_callback=mx.callback.Speedometer(batch_size, 50),)

