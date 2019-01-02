#!usr/bin/python
# -*- coding: utf-8 -*-
import random, time, queue
from copy import deepcopy, copy
from multiprocessing.managers import BaseManager
from collections import deque
import numpy as np
import csv
import tensorflow as tf

popsize=10 # popsize
ev_agent_1 = deque(maxlen=2000) # windows size
task_queue = queue.Queue(maxsize = popsize) # task queue;
result_queue = queue.Queue(maxsize = popsize)# result queue

def evolution_signal(update):
    """
    这是在TS-PBT第一阶段的执行函数,为了尽可能在分布式队列计算的框架下少做改动,
    仍旧在PBT的大框架进行单个个体的训练，这一过程就是单个agent的强化学习过程
    :param update: 
    :return: 
    """
    temp=[]
    temp.append(result.get())
    global_fitness = []
    global_hyper = []
    global_fitness.append(temp[0]['fitness'])
    global_hyper.append(temp[0]["hyperpara_sgd"])
    global_hyper.append(temp[0]["hyperpara_reward"])

    fitness = max(global_fitness)# acquire the max fitness
    print("The global_fitness is :",global_fitness)
    print("the global_hyperpara is:",global_hyper)
    #print("the global_weight is:", global_weight)
    global_fitness = np.array(global_fitness)
    index = np.argsort(global_fitness)# return the min -> max index
    ev_agent_1.extend(temp[0]["ev_record"])
    print("the ev is:",(sum(ev_agent_1))/len(ev_agent_1))
    print("the ev's length is:", len(ev_agent_1))
    if update <= 20 or ((sum(ev_agent_1))/len(ev_agent_1)) < 0.95:
        task.put(temp[0])
    elif ((sum(ev_agent_1))/len(ev_agent_1)) > 0.95:
        for i in range(popsize):
            task.put(temp[0])
    print("the task put ok")
    return (sum(ev_agent_1))/len(ev_agent_1)

def pbt(update):
    """
    pbt执行函数，此处我们优化6个DRL相关的超参数，具体参数baselines/baselines/a2c/a2c.py
    我们在a2c的model类中，定义了一些新的管道和操作，这也是PBT的代码层级的核心所在，请仔细阅读相关代码
    :param update: 
    :return: 
    """
    temp=[]
    for i in range(popsize):
        temp.append(result.get())

    global_fitness = []
    global_hyper = []
    #global_weight = []
    for i in range(popsize):
        global_fitness.append(temp[i]['fitness'])
        global_hyper.append(temp[i]["hyperpara_sgd"])
        global_hyper.append(temp[i]["hyperpara_reward"])

        #global_weight.append(temp[i]["weight"])
    fitness = max(global_fitness)# acquire the max fitness
    summary = tf.Summary(value=[
        tf.Summary.Value(tag="fitness_best", simple_value=fitness)
    ])
    #summary_writer.add_summary(summary, update)
    print("The global_fitness is :",global_fitness)
    print("the global_hyperpara is:",global_hyper)
    #print("the global_weight is:", global_weight)
    global_fitness = np.array(global_fitness)
    index = np.argsort(global_fitness)# return the min -> max index
    # pbt中后20%的个体是从前20%的个体中随机采样，由于我们的群体大小只有10，因此随机采样的意义不大
    # 因此在实际操作中，直接复制top20%的个体替换后20%的个体
    temp[index[0]] = deepcopy(temp[index[popsize-1]])
    temp[index[1]] = deepcopy(temp[index[popsize-2]])
    # tuning ent_coef, vf_coef, lr_coef
    for i in range(2):
        for j in range(3):
            if random.random() <= 0.5:
                temp[index[i]]['hyperpara_sgd'][j] *= 0.8
            else:
                temp[index[i]]['hyperpara_sgd'][j] *= 1.2
    # tuning alpha
    for i in range(2):
        if random.random() <= 0.5:
            temp[index[i]]['hyperpara_sgd'][3] *= 0.8
        else:
            temp[index[i]]['hyperpara_sgd'][3] *= 1.2
            if temp[index[i]]['hyperpara_sgd'][3] >= 0.99:
                temp[index[i]]['hyperpara_sgd'][3] = 0.99
            else:
                pass
    # tuning nsteps
    for i in range(2):
        if random.random() <= 0.5:
            temp[index[i]]["hyperpara_reward"][0] = int(temp[index[i]]["hyperpara_reward"][0] * 1.2)
        else:
            temp[index[i]]["hyperpara_reward"][0] = int(temp[index[i]]["hyperpara_reward"][0] * 0.8)
            if temp[index[i]]["hyperpara_reward"][0] <= 5:
                temp[index[i]]["hyperpara_reward"][0] = 5
            else:
                pass
    # tuning gamma
    for i in range(2):
        if random.random() <= 0.5:
            temp[index[i]]["hyperpara_reward"][1] *= 0.8
        else:
            temp[index[i]]["hyperpara_reward"][1] *= 1.2
            if temp[index[i]]["hyperpara_reward"][1] >= 0.99:
                temp[index[i]]["hyperpara_reward"][1] = 0.99
            else:
                pass
    # print new hyperpara
    new_global_hyperpara = []
    for i in range(popsize):
        new_global_hyperpara.append(temp[i]["hyperpara_sgd"])
        new_global_hyperpara.append(temp[i]["hyperpara_reward"])
        task.put(temp[i])
    print("new hyperpara:", new_global_hyperpara)

# 从BaseManager继承的QueueManager:
class QueueManager(BaseManager):
    pass

# 把两个Queue都注册到网络上, callable参数关联了Queue对象:
QueueManager.register('master_task', callable=lambda: task_queue)
QueueManager.register('worker_result', callable=lambda: result_queue)
# 绑定端口5000, 设置验证码'abc':
manager = QueueManager(address=('', 5000), authkey=b'abc')
# 启动Queue:
manager.start()
# 获得通过网络访问的Queue对象:
task = manager.master_task()
result = manager.worker_result()
task_init=[]
task.put(task_init)

# 进行第一个阶段的单agent的训练（单纯的单个体的强化学习过程）
update = 0
while (1):
    while(result.qsize()!=1):
        print("Now is the update is %d;the result num is :%d;the task num is :%d" % (update,result.qsize(),task.qsize()))
        time.sleep(2)
        if(result.qsize()==1):
            print("the result num is :%d;the task num is :%d" % (result.qsize(), task.qsize()))
            print("Now the result is full ,begin evolute")
            break
    update += 1 # sgd update
    ev_value = evolution_signal(update)
    if update > 20 and ev_value > 0.95:
        break
# 完成第一阶段的预训练

# 开始第二阶段的PBT训练
# 计算剩余计算代价,本实验室中单个个体的计算代价是40M，因此总计算代价是400M.
# 在PBT阶段，每个个体会在完成800000的step后，完成当前超参数，配置下的训练，将结果压入result队列
rest = int(((10*40000000-(update)*800000)/10)/800000)
# 进行PBT训练
for i in range(rest):
    while(~result.full()):
        print("Now is the update is %d;the result length is :%d;the task length is :%d" % (i,result.qsize(),task.qsize()))
        time.sleep(2)
        if(result.full()):
            print("the result length is :%d;the task length is :%d" % (result.qsize(), task.qsize()))
            print("Now the result is full ,begin evolute")
            break
    # 任务队列中的10个个体已经全部训练完成，并压入result队列中
    # 执行PBT操作
    pbt(rest)

manager.shutdown()
print('master exit.')


