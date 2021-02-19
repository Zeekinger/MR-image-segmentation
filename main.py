#!/usr/bin/env python
# coding=utf-8
from network import Net
from data_provider import DataProvider
from operation import Operation
from config import Configuration
import numpy as np
import datetime
# np.random.seed(datetime.datetime.now().second)
np.random.seed(2019)


if __name__ == '__main__':
    conf = Configuration()
    
    net = Net(conf)
    dp = DataProvider(conf)
    op = Operation(data_provider=dp, conf=conf)
    
    #op.train(net)
    #op.eval_with_volumes(net, evaluation=True, inference=False)
    op.eval_with_volumes(net, evaluation=False, inference=True)

