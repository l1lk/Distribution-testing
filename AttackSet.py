import os
import argparse
import tqdm
import shutil
import numpy as np

import ntpath
from PIL import Image
from keras.models import load_model
from keras import backend as K

import cal_mmd
from mutators import Mutators
from ImgMutators import Mutators as pixel_Mutators
from GA import Population


os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# 三个图像预处理函数
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


preprocess_dic = {
    'cifar': cifar_preprocessing,
    'mnist': mnist_preprocessing,
    'svhn': svhn_preprocessing,
    'fmnist': mnist_preprocessing
}


# 将seed转换为ga的输入： (100,28,28,3) --> (num,100,28,28,3)
def create_image_indvs_mmd(imgs, num): 
    '''
    imgs: input seed
    num: number of individuals
    indivs: ga的输入
    '''
    indivs = []
    if len(imgs.shape) < 4:
        imgs = imgs.reshape(-1, 28, 28, 1)
    shape = imgs.shape
    # 保留初始种子
    indivs.append(imgs)
    # 对初始种子中的每张图像做mutate，循环生成新的individual
    for i in range(num-1):
        tmp_indivs = []
        for img in imgs:
            tmp_indivs.append(Mutators.image_random_mutate_ga(img))
        indivs.append(np.array(tmp_indivs).reshape(shape))
    return indivs

# 对模型的一个输入，得到每层的输出
def predict(input_data, model):
    '''
    input_data: 模型输入数据
    model: 测试/攻击的模型
    '''
    inp = model.input
    layer_outputs = []
    for layer in model.layers[1:]:
        layer_outputs.append(layer.output)
    functor = K.function(inp, layer_outputs)
    outputs = functor([input_data])
    return outputs


# 获得ga突变操作的mutation function, mutation function定义在mutators.py
def build_mutate_func_mmd():
    def func(img):
        shape = img.shape
        return Mutators.image_random_mutate_ga(img).reshape(shape)
    return func


# 保存达到预期目标的样本
def build_save_func_mmd(npy_output, img_output, ground_truth, seedname, target): # crash_dir, img_dir, ground_truth, seed_name, target
    def save_func(indvs, round):
        prediction = []
        data = []
        probs = []
        
        for idv in indvs:
            prediction.append(idv[0])
            data.append(idv[1])
            probs.append(idv[2])
        # prediction = indvs[:,0]
        # data = indvs[:,1]
        # probs = indvs[:,2]
        

        for i, item in enumerate(prediction):
            #save image
            for img_idx in range(len(item)):
                name = "{}_{}_tar{}_gt{}_pred{}_{:.3f}".format(round, seedname,
                                                target, ground_truth,
                                                prediction[i][img_idx], probs[i][img_idx])
                
                np.save(os.path.join(npy_output, name + '.npy'), data[i])
        
                x = np.uint8(data[i][img_idx])
                shape = x.shape
                if shape[-1] == 1:
                    x = np.reshape(x, (shape[0], shape[1])) # shape[0], shape[1]
                img0 = Image.fromarray(x)
                img0.save(os.path.join(img_output, name + '.png'))
                # img0.save(os.path.join(img_output, name + '_image_' + str(i) + '_' + str(img_idx) + '.png'))

    return  save_func


def diff_object_func_v2(model, preprocess, label, nes, target_num = -1, threshold = 0.6, logits = True): # model, preprocess, ground_truth, target, threshold=ratio
    def func(indvs):

        array = []
    
        prediction = []
        wrong_pred_index = []
        
        evaluations = []
        prob_results = []

        out_predictions = []
        out_indvs = []
        mmds = []

        cluster_indexes = []

       
        for idx, ind in tqdm.tqdm(enumerate(indvs)): # for each ind
            cluster_indexes.append(0)
            # 模型每层的输出
            tmp_outputs = predict(preprocess(ind), model)
            # 模型预测结果
            tmp_prediction = np.argmax(tmp_outputs[-1], axis=1)
            # 被错误预测样本的index
            tmp_wrong_pred_index = np.nonzero(tmp_prediction != label)[0]

            # 是否使用logits层输出
            if logits:
                tmp_evaluations = tmp_outputs[-2]
            else:
                tmp_evaluations = tmp_outputs[-1]
                
            tmp_prob_results = tmp_outputs[-1]
            
            array.append(ind)
            wrong_pred_index.append(tmp_wrong_pred_index)
            evaluations.append(tmp_evaluations)
            prediction.append(tmp_prediction)
            prob_results.append(tmp_prob_results)
           
            # compute mmd
            mmds.append(cal_mmd.cal_mmd(evaluations[idx], nes).cpu().detach().numpy())

        interest_indexes = []
        interest_probs = []
        # 对每个预测错误的样本
        for idx, tmp_idx in enumerate(wrong_pred_index):
            tmp_interest_indexes = []
            tmp_interest_probs = []
            
            for i_idx in tmp_idx: # for some wrong prediction seeds(images) index in one cluster
                # 获得错误预测的置信度
                interest_prob = prob_results[idx][i_idx][prediction[idx][i_idx]] # For each image: [prob_0, prob_1, prob_2]
                # If it is a targeted configuration, we only care about our target goal.
                if target_num != -1 and prediction[idx][i_idx] != target_num:
                    continue
                
                # 如果置信度超过阈值，则认为成功
                if interest_prob > threshold:
                    
                    tmp_interest_indexes.append(i_idx)
                    tmp_interest_probs.append(interest_prob)
                    cluster_indexes[idx] = idx
            
            interest_indexes.append(tmp_interest_indexes)
            interest_probs.append(tmp_interest_probs)

            
        for idx in range(len(indvs)):
            out_predictions.append(prediction[idx][interest_indexes[idx]])
            out_indvs.append(array[idx][interest_indexes[idx]])
        
        return out_predictions, out_indvs, interest_probs, interest_indexes, cluster_indexes, mmds 

        # prediction[interest_indexes], array[interest_indexes], interest_probs, interest_indexes, fitness
    return func




if __name__ == '__main__':

    #=============================Initializing================================
    
    parser = argparse.ArgumentParser(description='coverage guided fuzzing')
    parser.add_argument('-i', help='seed path')
    parser.add_argument('-o', help='output path')
    parser.add_argument('-pop_num', help='number of individuals', type =int, default=1000)
    parser.add_argument('-type', help="dataset", choices=['mnist','imagenet','cifar','svhn', 'fmnist'],
                        default='mnist')
    parser.add_argument('-model_type', help='model',
                        choices=['lenet1', 'lenet5', 'resnet20', 'mobilenet', 'vgg16', 'resnet50'], default='lenet5')
    parser.add_argument('-model', help="model path")
    parser.add_argument('-ratio', type=float,help="threshold of confidence for saving samples", default=0) # e.g., 设置为0.4时，攻击成功的样本不仅要被模型错分，错分的置信度要超过0.4
    parser.add_argument('-subtotal', type=int, default=400)

    parser.add_argument('-timeout', help="setting the maximum runtime for GA", type=int, default=9999)
    parser.add_argument('-max_iteration', help="Setting the maximum iteration for GA", type=int, default=1000)
    parser.add_argument('-first_attack', choices=[0,1], type=int, default=0) 
    parser.add_argument('-target', help='target class for optimization', choices=[-1,0,1,2,3,4,5,6,7,8,9], type=int, default=-1)

    parser.add_argument('-mode', help='pixel level or image level mutation', choices=['pixel', 'image'], default='image')
    parser.add_argument('-logits', help='using logits or other layer', default=True)

    args = parser.parse_args()

    # 输入路径，文件名
    input_file =  args.i # "../seeds/mnist_seeds/0_0.npy"
    seed = ntpath.basename(input_file)
    seed_name = seed.split(".")[0]

    
    pop_num =  args.pop_num         #1000
    type_ =  args.type            #"mnist"
    model_type = args.model_type      # "lenet5"
    # 模型路径
    model = args.model             #"../models/lenet5.h5"

    # GA的超参数
    ratio = args.ratio             #0
    subtotal = args.subtotal          #400
    timeout = args.timeout           #30
    max_iteration = args.max_iteration   #1000
    first_attack =args.first_attack     #0
    target = args.target            #-1
    
    # 输出路径
    ga_output = "{0}/{1}_output_{2}".format(args.o, seed_name, target)
    

    #=============================Main Logic================================

    # Build directory structure
    if os.path.exists(ga_output):
        shutil.rmtree(ga_output)

    img_dir =  os.path.join(ga_output, 'imgs')
    crash_dir = os.path.join(ga_output, 'crashes')
    os.makedirs(crash_dir)
    os.makedirs(img_dir)
    

    # Load a model
    preprocess = preprocess_dic[type_] # 加载数据集对应的预处理函数
    model = load_model(model) # 如果是torch模型，需要修改
    
    # Load initial seed and give a prediction
    orig_imgs = np.load(input_file)
    
    ground_truth = np.argmax(model.predict(preprocess(orig_imgs))[0]) # 记录gtruth，由于我的seed中的图像都是来自相同类别的，所以只保留了一个
    
    # Load nes；nes中保存了target类别的样本的的特征，在之前实现的过程中，为了方便记录indiv距离所有类别的距离，这里加载时所有类别的样本都有加载
    if args.target != -1:
        nes_path = os.path.join(args.i.split("/")[0], args.i.split("/")[1], args.i.split("/")[2], args.i.split("/")[3])
        nes_pth_list = os.listdir(nes_path)
        nes_pth_list.sort()
    
    
    if nes_pth_list is not None:
        nes = []
        for p in nes_pth_list:
            print('ne path {}'.format(p))
            path = os.path.join(nes_path, p)
            tmp_nes = np.load(path)   
            nes_outputs = predict(preprocess(tmp_nes), model)
            nes.append(nes_outputs[-2])
    
    # seed -> individuals
    inds = create_image_indvs_mmd(orig_imgs, pop_num)


    
    # build a mutate function
    mutation_function = build_mutate_func_mmd()
    # build a save function
    save_function = build_save_func_mmd(crash_dir, img_dir, ground_truth, seed_name, target)
    # build a score function
    fitness_compute_function = diff_object_func_v2(model, preprocess, ground_truth, nes[target], target, threshold=ratio, logits=True)
    
    pop = Population(inds,mutation_function,fitness_compute_function,save_function,ground_truth,
                    subtotal=subtotal, max_time=timeout, seed=orig_imgs, max_iteration=max_iteration, mode=args.mode, model=model, nes=nes, pop_num=pop_num, target=target, type_=type_)
    pop.evolve_process(ga_output, target)


