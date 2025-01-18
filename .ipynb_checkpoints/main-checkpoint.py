import sys, os
import logging

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
from federated.serverbn import FedBN
from federated.serverpFedMe import pFedMe
from federated.serverrep import FedRep
from federated.model_trainer_segmentation import ModelTrainerSegmentation, ModelTrainerSegmentation_Rep, ModelTrainerSegmentation_pFedMe


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
        # 使用 ResnetUNet 模型（包括训练和 ood_test）
        from nets.models import ResnetUNet
        model = ResnetUNet(out_channels=2, resnet_type=args.resnet_type, pretrained=True)
        from federated.model_trainer_segmentation_MY import ModelTrainerSegmentation
        ModelTrainerClass = ModelTrainerSegmentation
    elif args.mode == "fedrep":
        from nets.models import UNetRep
        model = UNetRep(input_shape=[3, 384, 384])
        from federated.model_trainer_segmentation import ModelTrainerSegmentation_Rep
        ModelTrainerClass = ModelTrainerSegmentation_Rep
    elif args.mode == "pFedMe":
        from nets.models import UNet
        model = UNet(input_shape=[3, 384, 384])
        from federated.model_trainer_segmentation import ModelTrainerSegmentation_pFedMe
        ModelTrainerClass = ModelTrainerSegmentation_pFedMe
    else:
        # 使用 UNet 模型（包括训练和 ood_test）
        from nets.models import UNet
        model = UNet(input_shape=[3, 384, 384])
        from federated.model_trainer_segmentation import ModelTrainerSegmentation
        ModelTrainerClass = ModelTrainerSegmentation

    # 创建模型训练器
    model_trainer = ModelTrainerClass(model, args)

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
     if args.mode == "fediop":
          federated_api = FedIOP(datasets, device, args, model_trainer)         
     if args.mode == "fedavg":
          federated_api = FedAvg(datasets, device, args, model_trainer)   
     if args.mode == "fedbn":
          federated_api = FedBN(datasets, device, args, model_trainer)   
     if args.mode == "pFedMe":
          federated_api = pFedMe(datasets, device, args, model_trainer)   
     if args.mode == "fedrep":
          federated_api = FedRep(datasets, device, args, model_trainer)   
            
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
     # 如果 args.ood_test 为 True，则进行 OOD 测试     OOD测试是什么？怎么测试？
     if args.ood_test:
          #   打印提示信息，并导入 RouteConv2D 和 RouteConvTranspose2D 模块，这些模块可能包含动态路由机制的实现
          print('Test time dynamic route')
          from nets.routeconv import RouteConv2D, RouteConvTranspose2D
          # 定义一个字典 global_round，其中包含不同数据集（如I2CVB、ISBI等）在训练时保存的特定轮次的模型快照索引。
          global_round = {
               'I2CVB': 9,
               'ISBI': 99,
               'HK': 96,
               'BIDMC': 97,
               'UCL': 95,
               'ISBI_1.5': 99,
               }
          ckpt = torch.load('/root/autodl-tmp/pythonproject/IOP-FL-main/io-pfl-exp/checkpoint/{}/{}/seed{}_batch{}_lr{}/{}/fedala_global_round{}'.format(args.data,args.target,args.seed,args.batch,args.lr,args.mode,global_round[args.target]))


          model_trainer.set_model_params(ckpt)
          # 打印初始化完成信息。
          print('Finish intialization')
          # 定义一个字典 rounds，其中包含不同数据集在训练时保存的多个轮次的模型快照索引  什么是快照索引，有什么用
          rounds = {
               'I2CVB': [3, 9, 6, 2, 7],            
              # 'I2CVB': [55, 73, 89, 92, 79],
               'ISBI': [84, 87, 97, 99, 86],
               'HK': [80, 92, 73, 87, 91],
               'BIDMC': [78, 91, 72, 80, 74],
               'UCL': [99, 90, 87, 91, 99],
               'ISBI_1.5': [50, 83, 99, 99, 94],
               }
          # 加载多个特定轮次的模型快照路径
          paths = [
                            torch.load('/root/autodl-tmp/pythonproject/IOP-FL-main/io-pfl-exp/checkpoint/{}/{}/seed{}_batch{}_lr{}/{}/fedala_idx_0_round{}'.format(args.data, args.target, args.seed, args.batch, args.lr, args.mode, rounds[args.target][0])),
              torch.load('/root/autodl-tmp/pythonproject/IOP-FL-main/io-pfl-exp/checkpoint/{}/{}/seed{}_batch{}_lr{}/{}/fedala_idx_1_round{}'.format(args.data, args.target, args.seed, args.batch, args.lr, args.mode, rounds[args.target][1])),
              torch.load('/root/autodl-tmp/pythonproject/IOP-FL-main/io-pfl-exp/checkpoint/{}/{}/seed{}_batch{}_lr{}/{}/fedala_idx_2_round{}'.format(args.data, args.target, args.seed, args.batch, args.lr, args.mode, rounds[args.target][2])),
              torch.load('/root/autodl-tmp/pythonproject/IOP-FL-main/io-pfl-exp/checkpoint/{}/{}/seed{}_batch{}_lr{}/{}/fedala_idx_3_round{}'.format(args.data, args.target, args.seed, args.batch, args.lr, args.mode, rounds[args.target][3])),
              torch.load('/root/autodl-tmp/pythonproject/IOP-FL-main/io-pfl-exp/checkpoint/{}/{}/seed{}_batch{}_lr{}/{}/fedala_idx_4_round{}'.format(args.data, args.target, args.seed, args.batch, args.lr, args.mode, rounds[args.target][4])),
              torch.load('/root/autodl-tmp/pythonproject/IOP-FL-main/io-pfl-exp/checkpoint/{}/{}/seed{}_batch{}_lr{}/{}/fedala_global_round{}'.format(args.data,args.target,args.seed,args.batch,args.lr,args.mode,global_round[args.target]))
              
              # torch.load('/root/autodl-tmp/pythonproject/IOP-FL-main/io-pfl-exp/checkpoint/prostate/I2CVB/seed0_batch16_lr0.001/fedavg/fedavg_global_round99'),
              #  torch.load('/root/autodl-tmp/pythonproject/IOP-FL-main/io-pfl-exp/checkpoint/prostate/I2CVB/seed0_batch16_lr0.001/fedavg/fedavg_global_round99'),
              # torch.load('/root/autodl-tmp/pythonproject/IOP-FL-main/io-pfl-exp/checkpoint/prostate/I2CVB/seed0_batch16_lr0.001/fedavg/fedavg_global_round99'),
              # torch.load('/root/autodl-tmp/pythonproject/IOP-FL-main/io-pfl-exp/checkpoint/prostate/I2CVB/seed0_batch16_lr0.001/fedavg/fedavg_global_round99'),
              # torch.load('/root/autodl-tmp/pythonproject/IOP-FL-main/io-pfl-exp/checkpoint/prostate/I2CVB/seed0_batch16_lr0.001/fedavg/fedavg_global_round99'),
              # torch.load('/root/autodl-tmp/pythonproject/IOP-FL-main/io-pfl-exp/checkpoint/prostate/I2CVB/seed0_batch16_lr0.001/fedavg/fedavg_global_round99')
          ]

          # 遍历模型中的所有模块。
          # 如果模块是 RouteConv2D 或 RouteConvTranspose2D 类型，则调用其 _mix_trajectories 方法，将多个模型快照路径混合到当前模块中。
          for m in model_trainer.model.modules():
              if isinstance(m, RouteConv2D) or isinstance(m, RouteConvTranspose2D):
                  m._mix_trajectories(paths)
          # 调用 test_time_adaptation_by_iopfl 方法，在测试时进行模型自适应，返回测试指标
          metrics = federated_manager.ood_client.test_time_adaptation_by_iopfl(None)

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
     
     
