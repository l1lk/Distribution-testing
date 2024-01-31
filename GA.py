import sys
sys.path.append('.')
import numpy as np
import copy
import random
import os
import time
from ImgMutators import Mutators as pixel_Mutators
from mutators import Mutators
from tensorflow.keras import backend as K
import cal_mmd
import tqdm
import matplotlib.pyplot as plt


# 跟AttackSet.py里的一样
def create_image_indvs_mmd(imgs, num): # num * clusters, each cluster[n*images]  (done)
    indivs = []
    shape = imgs.shape
    if len(shape) < 4:
        shape = (shape[0], shape[1], shape[2], 1)
        imgs = np.array(imgs).reshape(shape)
    
    for i in range(num):
        tmp_indivs = []
        for img in imgs:
            tmp_indivs.append(Mutators.image_random_mutate_ga(img))
        indivs.append(np.array(tmp_indivs).reshape(shape)) 
    return indivs

def predict(input_data, model):
    inp = model.input
    layer_outputs = []
    for layer in model.layers[1:]:
        layer_outputs.append(layer.output)
    # functor = K.function([inp] + [K.learning_phase()], layer_outputs)
    functor = K.function(inp, layer_outputs)
    outputs = functor([input_data])
    return outputs

def mnist_preprocessing(x_test):
    temp = np.copy(x_test)
    temp = temp.reshape(temp.shape[0], 28, 28, 1)
    temp = temp.astype('float32')
    temp /= 255
    return temp

def cifar_preprocessing(x_test):
    temp = np.copy(x_test)
    temp = temp.astype('float32')
    mean = [125.307, 122.95, 113.865]
    std = [62.9932, 62.0887, 66.7048]
    for i in range(3):
        temp[:, :, :, i] = (temp[:, :, :, i] - mean[i]) / std[i]
    return temp

def svhn_preprocessing(x_test):
    temp = np.copy(x_test)
    temp = temp.astype('float32')
    temp /= 255.
    mean = [0.44154793, 0.44605806, 0.47180146]
    std = [0.20396256, 0.20805456, 0.20576045]
    for i in range(temp.shape[-1]):
        temp[:, :, :, i] = (temp[:, :, :, i] - mean[i]) / std[i]       
    return temp




class Population():

    def __init__(self, individuals, 
                 mutation_function, 
                 fitness_compute_function,
                 save_function,
                 groud_truth,
                 subtotal,  
                 seed,
                 max_iteration,
                 mode,
                 model,
                 nes,
                 pop_num,
                 target,
                 type_,
                 tour_size=20, cross_rate=0.5, mutate_rate=0.01, max_time=30):
        
        self.individuals = individuals # a list of individuals, current is numpy
        self.mutation_func = mutation_function
        self.fitness_fuc = fitness_compute_function
        self.save_function = save_function
        self.ground_truth = groud_truth
        self.subtotal = subtotal
        self.seed =seed
        self.first_iteration_used = max_iteration
        self.mode = mode
        self.model = model
        self.nes = nes
        self.pop_num = pop_num
        self.target = target
        self.type_ = type_

        
        self.tournament_size = tour_size
        self.cross_rate = cross_rate
        self.mutate_rate = mutate_rate
        self.first_time_attacked = max_time
        
        
        self.fitness = None   # a list of fitness values
        self.pop_size = len(self.individuals)
        self.order = []
        self.best_fitness = -1000
        self.best_mmds = 99
        self.min_idx = 0
        self.record_mmds = []
        self.mutator_idx = []
        self.round = 0
        self.seed_output = ''
        self.mmds = None
        
        # other info
        self.switch_counter = 0
        self.other_mmds = [[] for i in range(len(self.nes))]
        self.preprocess_dic = {
            'cifar': cifar_preprocessing,
            'mnist': mnist_preprocessing,
            'svhn': svhn_preprocessing,
            'fmnist': mnist_preprocessing
        }
        self.preprocess = self.preprocess_dic[self.type_]
    
    
    def evolve_process(self, seed_output, target):
        start_time = time.time()
        i = 0
        counter = 0
        success = 0
        plot_file = open(os.path.join(seed_output, 'plot.log'), 'a+')
        plot_file.write("This is for target %d(non-target:-1,target:0-9). \n"%target)
        self.seed_output = seed_output # ga的输出路径

        self.mmds = self.cal_cur_mmd(self.individuals, self.nes[int(target)]) # 初始化每个输入的mmd score
       
        while True:
            
            self.record_mmds.append(self.best_mmds) # best mmd to target 
            
            # 时间或轮次到预设值时停止
            if i > self.first_iteration_used:
                print("reach the max iteration")
                break
            if time.time()-start_time > self.first_time_attacked:
                print("reach the max time")
                break
            
            i += 1
            self.round = i
            results = self.evolvePopulation()
          
            cur_time = time.time()
            if results is None:
                print("Used time: {} ,Total generation: {}, best mmd: {:.3f}".format(cur_time-start_time,i, self.best_mmds))
                plot_file.write("Used time: {} ,Total generation: {}, best mmd: {:.3f} \n"
                                .format(cur_time-start_time,i, self.best_mmds))
            else:
                # results = np.array(results)
                cluster_prob_list = []
                cluster_list = []
                pred_list = []
                for res in results:
                    cluster_prob_list.append(res[-3])
                    cluster_list.append(res[-2])
                    pred_list.append(res[0])

                max_ = 0
                index_ = [0, 0]
                
                # print('prob {}'.format(results[:,-3]))
                for idx, cluster_prob in enumerate(cluster_prob_list):
                    tmp = max(cluster_prob)
                    if tmp > max_:
                        max_ = tmp
                        index_[0] = idx # which cluster
                        index_[1] = np.argmax(cluster_prob) # which specific image
                           
                # highest_prob = max(results[-2])  # interest_probs
                # index = np.argmax(results[-2])
                highest_prob = max_
                index = index_
                # print('results len: {} (GA.py line98)'.format(len(results)))
                # print('idx 1: {}'.format(index[1]))
                pred = pred_list[index[0]][index[1]]
                cur_mutator_num = 0
                for cluster in cluster_list:
                    cur_mutator_num += len(cluster)
                counter += cur_mutator_num
                print("Used time: {} , Total generation: {}, best mmd {:.3f}, find {} mutators, highest_prob is {:.3f}, prediction is {}".format(cur_time-start_time, i, self.best_mmds, cur_mutator_num, highest_prob, pred)) 
                plot_file.write("Used time: {}, Total generation: {}, best mmd {:.3f}, find {} mutators, highest_prob is {:.3f}, prediction is {} \n".format(cur_time-start_time, i, self.best_mmds, len(cluster_list), highest_prob, pred))
                # print("Used time:%.4f ,Total generation: %d, best fitness:%.9f, find %d mutators,highest_prob is %.5f,prediction is %d"%(cur_time-start_time,i, self.best_fitness,len(results[-1]),highest_prob,pred)) 
                # plot_file.write("Used time:%.4f ,Total generation: %d, best fitness:%.9f, find %d mutators,highest_prob is %.5f,prediction is %d \n"%(cur_time-start_time,i, self.best_fitness,len(results[-1]),highest_prob,pred))
                
                # 保存攻击成功的样本
                self.save_function(results, i)
                success = 1
                   
                
                wrong_pred_indexes = []
                for idx, re in enumerate(cluster_list):
                    wrong_pred_indexes.append(re)

                self.mutator_idx.append(i)
            
            if i % 100 == 0:
                plot_file.flush()
        
        whole_pth = seed_output.split('/')
        # print(whole_pth)
        chart_name = whole_pth[-1][6] + '_' + whole_pth[2]
        
        mmd_records_pth = seed_output + '/mmd_records/'
        if not os.path.exists(mmd_records_pth):
            os.makedirs(mmd_records_pth)
        
        with open(mmd_records_pth + chart_name + '_target_mmd_records.txt', 'w') as f:
            for record in self.record_mmds[1:]:
                f.write(str(record)+'\n')
        

        print('Success' if success else 'Failed')
        print("Total mutators:%d "%counter)
        plot_file.write("success:%d \n"%success)
        plot_file.write("Total mutators:%d \n"%counter)
        plot_file.flush()
        plot_file.close()
        
    def crossover(self, inds1, inds2): # inds1[N, 1, 28, 28, 1]
        # e.g. 两个（100，28，28，1）,crossover后新的individual每个图像随机来自输入的两个inds
        new_ind = []
        
        mask = np.random.uniform(0,1,len(inds1))
        mask[mask < self.cross_rate] = 0
        mask[mask >= self.cross_rate] = 1
        
        one = [int(x) for x in 1-mask]
        two = [int(x) for x in mask]
        
        for idx, i in enumerate(one):
            if i == 1:
                new_ind.append(inds1[idx])

        for idx, i in enumerate(two):
            if i == 1:
                new_ind.append(inds2[idx])

        return np.array(new_ind)


    def cal_cur_mmd(self, indvs, nes):
        mmds = []
        for idx, ind in tqdm.tqdm(enumerate(indvs)): 
            tmp_outputs = predict(self.preprocess(ind), self.model)
            mmds.append(cal_mmd.cal_mmd(tmp_outputs[-2], nes).cpu().detach().numpy())
        return mmds
    
    def cal_best_mmd(self, ind, nes):
        tmp_outputs = predict(self.preprocess(ind), self.model)
        return cal_mmd.cal_mmd(tmp_outputs[-2], nes).cpu().detach().numpy()


    def evolvePopulation(self):
        # 将individuals分组，分别crossover，mutation
        if self.pop_size % self.subtotal == 0:
            group_num = int(self.pop_size / self.subtotal) 
        else:
            group_num = int(self.pop_size / self.subtotal) + 1
        
        # Divide initial population into several small group and start a tournament
        sorted_mmds = []
        index_ranges = []
        prepare_list = locals()
        for i in range(group_num):
            if i != group_num-1:
                prepare_list['sorted_mmds_indexes_'+str(i+1)] = sorted(range(i*self.subtotal,(i+1)*self.subtotal), 
                                                       key=lambda k: self.mmds[k], reverse=False)
                index_ranges.append((i*self.subtotal,(i+1)*self.subtotal))
            else:
                prepare_list['sorted_mmds_indexes_'+str(i+1)] = sorted(range(i*self.subtotal,self.pop_size), 
                                                       key=lambda k: self.mmds[k], reverse=False)
                index_ranges.append((i*self.subtotal,self.pop_size))
                
            sorted_mmds.append(prepare_list['sorted_mmds_indexes_'+str(i+1)])
            
        new_indvs = []
        for j in range(group_num):
            sorted_mmds_indexes = sorted_mmds[j]
            best_index = sorted_mmds_indexes[0]   # min mmds
            # print('best_index: {}'.format(best_index))
            (start,end) = index_ranges[j]
            base_index = self.subtotal * j
            for i in range(start,end):
                item = self.individuals[i]
                if i == best_index:  # keep best
                    new_indvs.append(item)
                else:
                    # self.tournament_size should be smaller than end-start
                    order_seq1 = np.sort(np.random.choice(np.arange(start-base_index,end-base_index), self.tournament_size, replace=False))
                    order_seq2 = np.sort(np.random.choice(np.arange(start-base_index,end-base_index), self.tournament_size, replace=False))
                    # pick two best candidate from this tournament

                    first_individual = self.individuals[sorted_mmds_indexes[order_seq1[0]]]
                    second_individual = self.individuals[
                        sorted_mmds_indexes[order_seq2[0] if order_seq2[0] != order_seq1[0] else order_seq2[1]]]
                    
                    # crossover
                    ind = self.crossover(first_individual, second_individual)
                    
                    # mutation
                    if random.uniform(0, 1) < self.mutate_rate:
                        tmp_indivs = []
                        for i in ind:   
                            tmp_indivs.append(self.mutation_func(i))    
                        ind = np.array(tmp_indivs)                 
                    
                    new_indvs.append(ind)

        self.individuals = new_indvs
        
        # 计算新indivisuals是否攻击成功
        results = self.fitness_fuc(self.individuals)
        
        wrong_pred_indexes = results[-3]
        self.mmds = results[-1]
        
        # 更新最优mmd
        self.best_mmds = min(self.mmds)
        self.min_idx = np.argmin(self.mmds) 


        _result = results[:-1]
        tmp_results = []
        name = "{}_mmd{:.3f}".format(self.round, self.best_mmds)
        save_pth = self.seed_output + '/best_mmds'
        if not os.path.exists(save_pth):
            os.makedirs(save_pth)        
        np.save(os.path.join(save_pth, name + '.npy'), self.individuals[self.min_idx])
        
        # return 攻击成功的样本位置信息
        for idx, wrong_idx in enumerate(wrong_pred_indexes):
            
            # print('len(wrong_idx): {} (GA.py line172)'.format(len(wrong_idx)))
            if len(wrong_idx) > 0:
                res_list = []
                for res_idx in range(len(_result)):
                    res_list.append(_result[res_idx][idx])
                tmp_results.append(res_list)
            # print('len(tmp_results): {} (GA.py line172)'.format(len(tmp_results)))    
        if len(tmp_results) > 0:
            return tmp_results
        
        # Fail to find a mutator
        
        return None
