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
task_queue = queue.Queue(maxsize = popsize) # task queue;
result_queue = queue.Queue(maxsize = popsize)# result queue
# 记录文件
out1 = open('ga_fitness_1.csv','a',newline='')
csv_write1 = csv.writer(out1,dialect='excel') # use for plot
out2 = open("ga_hyperpara_1.csv","a",newline='')
csv_write2 = csv.writer(out2,dialect='excel')

out3 = open('ga_fitness_2.csv','a',newline='')
csv_write3 = csv.writer(out3,dialect='excel') # use for plot
out4 = open("ga_hyperpara_2.csv","a",newline='')
csv_write4 = csv.writer(out4,dialect='excel')

out5 = open('ga_fitness_3.csv','a',newline='')
csv_write5 = csv.writer(out5,dialect='excel') # use for plot
out6 = open("ga_hyperpara_3.csv","a",newline='')
csv_write6 = csv.writer(out6,dialect='excel')

out7 = open('ga_fitness_4.csv','a',newline='')
csv_write7 = csv.writer(out7,dialect='excel') # use for plot
out8 = open("ga_hyperpara_4.csv","a",newline='')
csv_write8 = csv.writer(out8,dialect='excel')

out9 = open("ga_fitness_5.csv","a",newline='')
csv_write9 = csv.writer(out9,dialect='excel')
out10 = open("ga_hyperpara_5.csv","a",newline='')
csv_write10 = csv.writer(out10,dialect='excel')
#--------------------------------
ev_agent_1 = deque(maxlen=2000) # windows size
#---------------------------------

def evolution_signal(update):
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
    csv_write1.writerow(temp[0]["episode_record"])# record the best episode
    csv_write3.writerow(temp[0]["episode_record"])# record the best episode
    csv_write5.writerow(temp[0]["episode_record"])# record the best episode
    csv_write7.writerow(temp[0]["episode_record"])# record the best episode
    csv_write9.writerow(temp[0]["episode_record"])# record the best episode

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


def PBT(update):
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
    # record the fitness
    csv_write1.writerow(temp[index[popsize-1]]["episode_record"])# record the best episode
    csv_write3.writerow(temp[index[popsize-2]]["episode_record"])# record the best episode
    csv_write5.writerow(temp[index[popsize-3]]["episode_record"])# record the best episode
    csv_write7.writerow(temp[index[popsize-4]]["episode_record"])# record the best episode
    csv_write9.writerow(temp[index[popsize-5]]["episode_record"])# record the best episode
    # record the hyperparameter
    csv_write2.writerow([temp[index[popsize-1]]["hyperpara_sgd"],temp[index[popsize-1]]["hyperpara_reward"]])
    csv_write4.writerow([temp[index[popsize-2]]["hyperpara_sgd"],temp[index[popsize-2]]["hyperpara_reward"]])
    csv_write6.writerow([temp[index[popsize-3]]["hyperpara_sgd"],temp[index[popsize-3]]["hyperpara_reward"]])
    csv_write8.writerow([temp[index[popsize-4]]["hyperpara_sgd"],temp[index[popsize-4]]["hyperpara_reward"]])
    csv_write10.writerow([temp[index[popsize-5]]["hyperpara_sgd"],temp[index[popsize-5]]["hyperpara_reward"]])

    # new pbt operation
    # deepcopy the weight and hyperpara
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

# # 2000 moving_average,20 times
# for update in range(20):
#     while(result.qsize()!=1):
#         print("Now is the update is %d;the result num is :%d;the task num is :%d" % (update,result_signal.qsize(),task_signal.qsize()))
#         time.sleep(2)
#         if(result.qsize()==1):
#             print("the result num is :%d;the task num is :%d" % (result_signal.qsize(), task_signal.qsize()))
#             print("Now the result is full ,begin evolute")
#             break
#     evolution_signal(update)

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


rest = int(((10*40000000-(update)*800000)/10)/800000)

for i in range(rest):
    while(~result.full()):
        print("rest is:", rest)
        print("Now is the update is %d;the result num is :%d;the task num is :%d" % (i,result.qsize(),task.qsize()))
        time.sleep(2)
        if(result.full()):
            print("the result num is :%d;the task num is :%d" % (result.qsize(), task.qsize()))
            print("Now the result is full ,begin evolute")
            break
    PBT(rest)

manager.shutdown()
print('master exit.')


