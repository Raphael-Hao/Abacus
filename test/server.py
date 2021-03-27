#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: raphael hao

import random


from abacus.server import Query, AbacusServer, Scheduler
from abacus.option import parse_options

run_config = parse_options()


def prepare_test_queries(total_queries=1000):
    test_queries = []
    for i in range(total_queries):
        model_id = i + 3
        # model_id = random.choice(run_config.serve_combination)
        bs = random.choice(run_config.supported_batchsize)
        seq_len = random.choice(run_config.supported_seqlen) if model_id == 6 else 0
        print(
            "{} selected model: {}, bs: {}, seq_len: {}".format(
                i, model_id, bs, seq_len
            )
        )
        test_queries.append(Query(model_id, bs, seq_len))
    for query in test_queries:
        query.set_op_pos(-1)
    return tuple(test_queries)


class TestScheduler:
    def test_query_feature(self):
        qos_query, l_query, m_query, r_query = prepare_test_queries(4)
        scheduler = Scheduler(
            run_config=run_config, barrier=None, queues=None, pipes=None, stop_flag=None
        )
        query_feature = scheduler.get_query_feature(
            4, qos_query, l_query=l_query, m_query=m_query, r_query=r_query
        )
        print(query_feature)
        r_query = None
        query_feature = scheduler.get_query_feature(
            3, qos_query, l_query=l_query, m_query=m_query, r_query=r_query
        )
        print(query_feature)
        m_query = None
        query_feature = scheduler.get_query_feature(
            2, qos_query, l_query=l_query, m_query=m_query, r_query=r_query
        )
        print(query_feature)
        l_query = None
        query_feature = scheduler.get_query_feature(
            2, qos_query, l_query=l_query, m_query=m_query, r_query=r_query
        )
        print(query_feature)
