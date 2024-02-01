import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import sys
import argparse
import tqdm
import shutil
import numpy as np
import torch
import ntpath
from PIL import Image
import cal_mmd
from mutators import Mutators
from GA import Population
from loguru import logger

level = 'INFO' # 'INFO'
logger.configure(handlers=[{"sink": sys.stderr, "level": level}])

def mnist_preprocessing_torch(x_test):
    temp = np.copy(x_test)
    temp = temp.reshape(temp.shape[0], 28, 28, 1)
    temp = np.transpose(temp, (0,3,1,2))
    temp = temp.astype('float32')
    temp /= 255
    temp = torch.tensor(temp, dtype=torch.float32)
    return temp

def own_data_preprocessing(x_input):
    pass

preprocess_dic = {
    'mnist': mnist_preprocessing_torch,
    'own': own_data_preprocessing
}

def get_layer_outputs(model, input_data):
    model_device = next(model.parameters()).device
    logger.debug('model device: {}', model_device)

    # input_data = input_data.reshape((input_data.shape[0], 1, 32, 32))
    #print(input_data.shape)
    # input_tensor = torch.from_numpy(input_data).float()
    # input_tensor = input_tensor.to(model_device)
    input_tensor = input_data
    layer_outputs = [] # cpu here -> not right
    
    def hook_fn(module, input, output):
        # layer_outputs.append(output.cpu().numpy())
        layer_outputs.append(output)

    hooks = []
    for layer in model.children():
        hook = layer.register_forward_hook(hook_fn)
        hooks.append(hook)

    # model = model.cuda()
    model.eval()

    with torch.no_grad():
        #input_tensor = input_tensor.cuda()
        model(input_tensor)

    for hook in hooks:
        hook.remove()

    return layer_outputs

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
        # print(data[0].shape)

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
    return  save_func


def diff_object_func_v2(model, preprocess, label, nes, target_num = -1, threshold = 0.6): # model, preprocess, ground_truth, target, threshold=ratio
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

        ind_num = len(indvs)
        concated_indvs = np.concatenate(indvs, axis=0)
        preprocessed_indvs = preprocess(concated_indvs).to(device)
        layer_outputs = get_layer_outputs(model, preprocessed_indvs)
        logits = layer_outputs[-2]
        softmax = layer_outputs[-1]
        preds = torch.argmax(layer_outputs[-1], dim=1)
        logits_length = logits.shape[-1]
        indvs_logits = logits.view(ind_num, len(logits)//ind_num, logits_length)
        indvs_softmax = softmax.view(ind_num, len(logits)//ind_num, logits_length)
        indvs_preds = preds.view(ind_num, len(logits)//ind_num)
        # print(indvs_preds)
        # print('test',indvs_logits.shape)
        for idx, ind_lgts in tqdm.tqdm(enumerate(indvs_logits)): # for each ind
            cluster_indexes.append(0)
            ind_preds = indvs_preds[idx]
            ind_softmax = indvs_softmax[idx]
            # 被错误预测样本的index
            tmp_wrong_pred_index = np.nonzero(torch.ne(ind_preds, label).detach().cpu().numpy())[0]
            
            # compute mmd
            mmds.append(cal_mmd.cal_mmd(ind_lgts, nes).cpu().detach().numpy())
            array.append(indvs[idx])
            wrong_pred_index.append(tmp_wrong_pred_index)
            evaluations.append(ind_lgts.detach().cpu().numpy())
            prediction.append(ind_preds)
            prob_results.append(ind_softmax)

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
                    tmp_interest_probs.append(interest_prob.item())
                    cluster_indexes[idx] = idx
            
            interest_indexes.append(tmp_interest_indexes)
            interest_probs.append(tmp_interest_probs)

        for idx in range(len(indvs)):
            out_predictions.append(prediction[idx][interest_indexes[idx]])
            out_indvs.append(array[idx][interest_indexes[idx]])
        
        return out_predictions, out_indvs, interest_probs, interest_indexes, cluster_indexes, mmds 

    #     # prediction[interest_indexes], array[interest_indexes], interest_probs, interest_indexes, fitness
    return func




if __name__ == '__main__':

    #=============================Initializing================================
    
    parser = argparse.ArgumentParser(description='coverage guided fuzzing')
    parser.add_argument('-seeds', help='seed path')
    parser.add_argument('-gt', help='ground_truth')
    parser.add_argument('-o', help='output path')
    parser.add_argument('-pop_num', help='number of individuals', type =int, default=1000)
    parser.add_argument('-type', help="dataset", choices=['mnist','imagenet','cifar','svhn', 'fmnist'，'own'],
                        default='mnist')
    parser.add_argument('-model_type', help='model',
                        choices=['lenet1', 'lenet5', 'resnet20', 'mobilenet', 'vgg16', 'resnet50'], default='lenet5')
    parser.add_argument('-model', help="model path")
    parser.add_argument('-ratio', type=float,help="threshold of confidence for saving samples", default=0) # e.g., 设置为0.4时，攻击成功的样本不仅要被模型错分，错分的置信度要超过0.4
    parser.add_argument('-subtotal', type=int, default=400)

    parser.add_argument('-timeout', help="setting the maximum runtime for GA", type=int, default=9999)
    parser.add_argument('-max_iteration', help="Setting the maximum iteration for GA", type=int, default=1000)
    parser.add_argument('-first_attack', choices=[0,1], type=int, default=0) 
    parser.add_argument('-target', help='target file path for optimization')
    parser.add_argument('-target_class', type=int, help='class index to optimize')

    parser.add_argument('-mode', help='pixel level or image level mutation', choices=['pixel', 'image'], default='image')

    args = parser.parse_args()

    
    device='cuda'
    # 输入路径，文件名
    input_file =  args.seeds # "../seeds/mnist_seeds/0_0.npy"
    gt = args.gt
    seed = ntpath.basename(input_file)
    seed_name = seed.split(".")[0]

    
    pop_num =  args.pop_num         #1000
    type_ =  args.type            #"mnist"
    model_type = args.model_type      # "lenet5"
    # 模型路径
    model_dir = args.model             #"../models/lenet5.h5"

    # GA的超参数
    ratio = args.ratio             #0
    subtotal = args.subtotal          #400
    timeout = args.timeout           #30
    max_iteration = args.max_iteration   #1000
    first_attack =args.first_attack     #0
    target = args.target            #-1
    target_class = args.target_class


    # # 输出路径
    ga_output = args.o
    

    #=============================Main Logic================================

    # Build directory structure
    if os.path.exists(ga_output):
        shutil.rmtree(ga_output)

    img_dir =  os.path.join(ga_output, 'imgs')
    crash_dir = os.path.join(ga_output, 'crashes')
    os.makedirs(crash_dir)
    os.makedirs(img_dir)
    

    # Load a model
    '''
    这里需要修改：模型和数据预处理，换到目标检测，模型输出对应分类的logits
    '''
    
    model = torch.load(model_dir).to(device)
    model.eval()
    preprocess = preprocess_dic[type_] # 加载数据集对应的预处理函数

    orig_imgs = np.load(input_file)
    ground_truth = torch.tensor(np.load(gt)).to(device)
    truth_idx = ground_truth[0].item()

    target_imgs = np.load(target)

    processed_target_imgs = preprocess(target_imgs).to(device)
    nes = get_layer_outputs(model, processed_target_imgs)[-2]
    
    # seed -> individuals
    inds = create_image_indvs_mmd(orig_imgs, pop_num)


    
    # build a mutate function
    mutation_function = build_mutate_func_mmd()
    # build a save function
    save_function = build_save_func_mmd(crash_dir, img_dir, truth_idx, seed_name, target_class)
    # build a score function
    fitness_compute_function = diff_object_func_v2(model, preprocess, ground_truth, nes, target_class, threshold=ratio)
    
    pop = Population(individuals=inds,
                     preprocess_function=preprocess,
                     get_layer_outputs=get_layer_outputs,
                     mutation_function=mutation_function,
                     fitness_compute_function=fitness_compute_function,
                     save_function=save_function,
                     groud_truth=ground_truth,
                     subtotal=subtotal, 
                     max_time=timeout, 
                     seed=orig_imgs, 
                     max_iteration=max_iteration, 
                     mode=args.mode, 
                     model=model, 
                     nes=nes, 
                     pop_num=pop_num, 
                     target=target, 
                     type_=type_, 
                     device=device)
    pop.evolve_process(ga_output, target)


