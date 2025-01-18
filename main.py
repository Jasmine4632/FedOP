import sys, os
import logging
import time 
from torch.utils import data
from torch.utils.data import dataset
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_path)
import numpy as np
import random
import torch
torch.cuda.empty_cache()

import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

from federated.configs import set_configs
from data.prostate.generate_data import load_partition_data_federated_prostate
from federated.serverMY import FedMY
from federated.model_trainer_segmentation_MY import ModelTrainerSegmentation
from federated.serveriop import FedIOP
from federated.serveravg import FedAvg
from federated.serverapple import APPLEAPI
from federated.serverfedfed import FedFed
from federated.serverper import PerFedAvg
from federated.serverditto import Ditto
from federated.serverprox import FedProx
from federated.servercp import FedCP
from federated.serverdg import FedDG
from federated.model_trainer_segmentation import ModelTrainerSegmentation,ModelTrainerSegmentationDitto,ModelTrainerSegmentationPer


# 设置确定性的随机数种子，确保实验的可重复性
def deterministic(seed):
     cudnn.benchmark = False
     cudnn.deterministic = True
     np.random.seed(seed)
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     random.seed(seed)
     
# 设置实验结果保存的路径
def set_paths(args):
     args.save_path = '/root/autodl-tmp/pythonproject/IOP-FL-main/io-pfl-exp/checkpoint/{}/{}/seed{}_batch{}_lr{}'.format(args.data, args.target, args.seed, args.batch, args.lr)
     exp_folder = '{}'.format(args.mode)

     if args.balance:
          exp_folder = exp_folder + '_balanced'

     print(exp_folder)
     args.save_path = os.path.join(args.save_path, exp_folder)
     if not os.path.exists(args.save_path):
          os.makedirs(args.save_path)

# 自定义模型训练器 根据参数 args 创建模型训练器        模型训练器是干什么，怎么训练的？
def custom_model_trainer(args):
    args.lr = 1e-3

    # 根据模式选择模型和模型训练器
    if args.mode == "fedMY":
        # from nets.ResnetUNet import ResnetUNet
        from nets.models import UNet
        # model = ResnetUNet(out_channels=2, resnet_type=args.resnet_type, pretrained=True)
        model = UNet(input_shape=[3, 384, 384])
        from federated.model_trainer_segmentation_MY import ModelTrainerSegmentation
        ModelTrainerClass = ModelTrainerSegmentation
    elif args.mode == "ditto":
        from nets.models import UNet
        model = UNet(input_shape=[3, 384, 384])
        from federated.model_trainer_segmentation import ModelTrainerSegmentationDitto
        ModelTrainerClass = ModelTrainerSegmentationDitto
    elif args.mode == "perfedavg":
        from nets.models import UNet
        model = UNet(input_shape=[3, 384, 384])
        from federated.model_trainer_segmentation import ModelTrainerSegmentationPer
        ModelTrainerClass = ModelTrainerSegmentationPer
    elif args.mode == "fedcp":
        from nets.models import UNet
        model = UNet(input_shape=[3, 384, 384])
        from federated.model_trainer_segmentation import ModelTrainerSegmentation
        ModelTrainerClass = ModelTrainerSegmentation
    elif args.mode == "feddg":
        from nets.models import UNet
        model = UNet(input_shape=[3, 384, 384])
        from federated.model_trainer_segmentation import ModelTrainerSegmentation
        ModelTrainerClass = ModelTrainerSegmentation
    else:
        # 使用 UNet 模型（包括训练和 ood_test）
        from nets.models import UNet
        model = UNet(input_shape=[3, 384, 384])
        from federated.model_trainer_segmentation import ModelTrainerSegmentation
        ModelTrainerClass = ModelTrainerSegmentation

    # 创建模型训练器
    model_trainer = ModelTrainerClass(model, args)
    print(f"model_trainer type: {type(model_trainer)}") 
    # 返回模型训练器
    return model_trainer

# 根据 args.data 加载相应的数据集
def custom_dataset(args):
     if args.data == "prostate":
          datasets = load_partition_data_federated_prostate(args)
     return datasets

# 创建联邦学习 API    API具体怎么运行的，代码在哪里实现什么功能   这里可以选择算法fedavg，fedala
def custom_federated_api(args, model_trainer, datasets):
     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
     # print(device)
     if args.mode == "fedMY":
          federated_api = FedMY(datasets, device, args, model_trainer)
     elif args.mode == "fediop":
          federated_api = FedIOP(datasets, device, args, model_trainer)         
     elif args.mode == "fedavg":
          federated_api = FedAvg(datasets, device, args, model_trainer)   
     elif args.mode == "apple":
          federated_api = APPLEAPI(datasets, device, args, model_trainer)   
     elif args.mode == "fedfed":
          federated_api = FedFed(datasets, device, args, model_trainer)   
     elif args.mode == "perfedavg":
          federated_api = PerFedAvg(datasets, device, args, model_trainer)   
     elif args.mode == "ditto":
          federated_api = Ditto(datasets, device, args, model_trainer) 
     elif args.mode == "fedcp":
          federated_api = FedCP(datasets, device, args, model_trainer) 
     elif args.mode == "fedprox":
          federated_api = FedProx(datasets, device, args, model_trainer) 
     elif args.mode == "feddg":
          federated_api = FedDG(datasets, device, args, model_trainer) 
     else:
          raise ValueError(f"Unknown mode: {args.mode}")  # 如果没有匹配到任何模式，则抛出异常       
     return federated_api
     

if __name__ == "__main__":
     args = set_configs()
     print(f"Layer Index from argparse: {args.layer_idx}")  # 打印解析出的 layer_idx
     args.generalize = False
     deterministic(args.seed)
     set_paths(args)
     os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
     log_path = args.save_path.replace('checkpoint', 'log')
     if not os.path.exists(log_path): os.makedirs(log_path)
     log_path = log_path+'/log.txt' if args.log else './log.txt'
     logging.basicConfig(filename=log_path, level=logging.INFO,
                    format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
     logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
     logging.info(str(args))

     # 使用custom_model_trainer函数创建模型训练器
     model_trainer = custom_model_trainer(args)
     # 使用custom_dataset函数创建模型训练器
     datasets = custom_dataset(args)
     # 创建联邦学习 API 并开始训练或测试
     federated_manager = custom_federated_api(args, model_trainer, datasets)
    
            
     if args.ood_test:
         if args.mode =='fedMY':
             start_time = time.time()
             # 动态路由机制逻辑
             print('Test time dynamic route')
             from nets.routeconv import RouteConv2D, RouteConvTranspose2D

             ckpt = torch.load('/root/autodl-tmp/pythonproject/IOP-FL-main/io-pfl-exp/checkpoint/{}/{}/seed{}_batch{}_lr{}/fedMY/{}_global_round40'.format(args.data,args.target,args.seed,args.batch,args.lr,args.mode))

             model_trainer.set_model_params(ckpt)
             # 打印初始化完成信息。
             print('Finish intialization')

             # 加载多个特定轮次的模型快照路径
             paths = [
                               torch.load('/root/autodl-tmp/pythonproject/IOP-FL-main/io-pfl-exp/checkpoint/{}/{}/seed{}_batch{}_lr{}/fedMY/{}_idx_0_round99'.format(args.data, args.target, args.seed, args.batch, args.lr,args.mode)),
                 torch.load('/root/autodl-tmp/pythonproject/IOP-FL-main/io-pfl-exp/checkpoint/{}/{}/seed{}_batch{}_lr{}/fedMY/{}_idx_1_round99'.format(args.data, args.target, args.seed, args.batch, args.lr,args.mode)),
                 torch.load('/root/autodl-tmp/pythonproject/IOP-FL-main/io-pfl-exp/checkpoint/{}/{}/seed{}_batch{}_lr{}/fedMY/{}_idx_2_round99'.format(args.data, args.target, args.seed, args.batch, args.lr,args.mode)),
                 torch.load('/root/autodl-tmp/pythonproject/IOP-FL-main/io-pfl-exp/checkpoint/{}/{}/seed{}_batch{}_lr{}/fedMY/{}_idx_3_round99'.format(args.data, args.target, args.seed, args.batch, args.lr,args.mode)),
                 torch.load('/root/autodl-tmp/pythonproject/IOP-FL-main/io-pfl-exp/checkpoint/{}/{}/seed{}_batch{}_lr{}/fedMY/{}_idx_4_round99'.format(args.data, args.target, args.seed, args.batch, args.lr,args.mode)),
                 torch.load('/root/autodl-tmp/pythonproject/IOP-FL-main/io-pfl-exp/checkpoint/{}/{}/seed{}_batch{}_lr{}/fedMY/{}_global_round99'.format(args.data,args.target,args.seed,args.batch,args.lr,args.mode))
             ]

             # 遍历模型中的所有模块。
             # 如果模块是 RouteConv2D 或 RouteConvTranspose2D 类型，则调用其 _mix_trajectories 方法，将多个模型快照路径混合到当前模块中。
             for m in model_trainer.model.modules():
                 if isinstance(m, RouteConv2D) or isinstance(m, RouteConvTranspose2D):
                     m._mix_trajectories(paths)
             # 调用 test_time_adaptation_by_iopfl 方法，在测试时进行模型自适应，返回测试指标
             metrics = federated_manager.ood_client.test_time_adaptation_by_iopfl(None)
             end_time = time.time()
             total_time = end_time - start_time  # 计算总时间        
             print(f"Total time for OOD test: {total_time:.2f} seconds")
                
         elif args.mode =='fediop':
             start_time = time.time()
             # 动态路由机制逻辑
             print('Test time dynamic route')
             from nets.routeconv import RouteConv2D, RouteConvTranspose2D

             global_round = {
                  'austin': 95,
                  'chicago': 99,
                  'kitsap': 96,
                  'massachusetts': 97,
                  'tyrol': 95,
                  'vienna': 99,
                  }
             ckpt = torch.load('/root/autodl-tmp/pythonproject/IOP-FL-main/io-pfl-exp/checkpoint/{}/{}/seed{}_batch{}_lr{}/{}/{}_global_round{}'.format(args.data,args.target,args.seed,args.batch,args.lr,args.mode,args.mode,global_round[args.target]))

             model_trainer.set_model_params(ckpt)
             # 打印初始化完成信息。
             print('Finish intialization')
             # 定义一个字典 rounds，其中包含不同数据集在训练时保存的多个轮次的模型快照索引  什么是快照索引，有什么用
             rounds = {
                  # 'austin': [99, 99, 99, 99, 99],            
                 'austin': [55, 73, 89, 92, 79],
                  'chicago': [84, 87, 97, 99, 86],
                  'kitsap': [80, 92, 73, 87, 91],
                  'massachusetts': [78, 91, 72, 80, 74],
                  'tyrol': [99, 90, 87, 91, 99],
                  'vienna': [50, 83, 99, 99, 94],
                  }
             # 加载多个特定轮次的模型快照路径
             paths = [
                               torch.load('/root/autodl-tmp/pythonproject/IOP-FL-main/io-pfl-exp/checkpoint/{}/{}/seed{}_batch{}_lr{}/fediop/{}_idx_0_round{}'.format(args.data, args.target, args.seed, args.batch, args.lr,args.mode, rounds[args.target][0])),
                 torch.load('/root/autodl-tmp/pythonproject/IOP-FL-main/io-pfl-exp/checkpoint/{}/{}/seed{}_batch{}_lr{}/fediop/{}_idx_1_round{}'.format(args.data, args.target, args.seed, args.batch, args.lr,args.mode, rounds[args.target][1])),
                 torch.load('/root/autodl-tmp/pythonproject/IOP-FL-main/io-pfl-exp/checkpoint/{}/{}/seed{}_batch{}_lr{}/fediop/{}_idx_2_round{}'.format(args.data, args.target, args.seed, args.batch, args.lr,args.mode, rounds[args.target][2])),
                 torch.load('/root/autodl-tmp/pythonproject/IOP-FL-main/io-pfl-exp/checkpoint/{}/{}/seed{}_batch{}_lr{}/fediop/{}_idx_3_round{}'.format(args.data, args.target, args.seed, args.batch, args.lr,args.mode, rounds[args.target][3])),
                 torch.load('/root/autodl-tmp/pythonproject/IOP-FL-main/io-pfl-exp/checkpoint/{}/{}/seed{}_batch{}_lr{}/fediop/{}_idx_4_round{}'.format(args.data, args.target, args.seed, args.batch, args.lr,args.mode, rounds[args.target][4])),
                 torch.load('/root/autodl-tmp/pythonproject/IOP-FL-main/io-pfl-exp/checkpoint/{}/{}/seed{}_batch{}_lr{}/fediop/{}_global_round{}'.format(args.data,args.target,args.seed,args.batch,args.lr,args.mode,global_round[args.target]))

             ]

             # 遍历模型中的所有模块。
             # 如果模块是 RouteConv2D 或 RouteConvTranspose2D 类型，则调用其 _mix_trajectories 方法，将多个模型快照路径混合到当前模块中。
             for m in model_trainer.model.modules():
                 if isinstance(m, RouteConv2D) or isinstance(m, RouteConvTranspose2D):
                     m._mix_trajectories(paths)
             # 调用 test_time_adaptation_by_iopfl 方法，在测试时进行模型自适应，返回测试指标
             metrics = federated_manager.ood_client.test_time_adaptation_by_iopfl(None)  
             end_time = time.time()
             total_time = end_time - start_time  # 计算总时间        
             print(f"Total time for OOD test: {total_time:.2f} seconds")
            
         elif args.mode in ['fedavg', 'fedcp', 'feddg', 'ditto', 'fedfed', 'fedprox', 'perfedavg']:
             start_time = time.time()
             print("执行 xxx 模式下的 OOD 测试...")

             ckpt = torch.load('/root/autodl-tmp/pythonproject/IOP-FL-main/io-pfl-exp/checkpoint/{}/{}/seed{}_batch{}_lr{}/{}/{}_global_round99'.format(args.data,args.target,args.seed,args.batch,args.lr,args.mode,args.mode))

             w_global = ckpt
             # 执行 OOD 测试
             metrics = federated_manager._ood_test_on_global_model(round_idx=args.comm_round - 1,ood_client=federated_manager.ood_client,ood_data=federated_manager.ood_data,w_global=w_global)
             end_time = time.time()
             total_time = end_time - start_time  # 计算总时间        
             print(f"Total time for OOD test: {total_time:.2f} seconds")

     # 如果 args.test 为 True，则加载模型进行测试   怎么测试的？ 
     elif args.test:
            ckpt = torch.load('/root/autodl-tmp/pythonproject/IOP-FL-main/io-pfl-exp/checkpoint/{}/{}/seed{}_batch{}_lr{}/{}/{}_global_round{}'.format(args.data, args.target, args.seed, args.batch, args.lr, args.mode, args.mode, args.comm_round - 1))
            model_trainer.set_model_params(ckpt)  
            test_data_local_dict = datasets[1][-2]
            # test the trajectroy on all local clients
            for client_idx in range(datasets[0]):
                 metrics = model_trainer.test(test_data_local_dict[client_idx], federated_manager.device, args)      #转到ModelTrainerSegmentation的test
                 print(metrics)
     # 否则，开始训练模型
     else:
          federated_manager.train()
     
     
