import argparse
import yaml

prosate = ['austin', 'chicago', 'kitsap', 'massachusetts', 'tyrol','vienna', 'whu', 'None']
available_datasets = prosate 


# set_configs 函数设置命令行参数，用于配置实验。
# 返回解析后的参数对象 args。
def set_configs():
     parser = argparse.ArgumentParser()
     parser.add_argument('--log', action='store_true', help='whether to log')
     parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
     parser.add_argument('--early', action='store_true', help='early stop w/o improvement over 10 epochs')
     parser.add_argument('--resnet_type', type=str, default='resnet34', choices=['resnet18', 'resnet34', 'resnet50', 'resnet101'], help='Specify the ResNet model type to use for U-Net encoder')
     parser.add_argument('--batch', type = int, default=16, help ='batch size')
     parser.add_argument("--source", choices=available_datasets, help="Source", nargs='+')
     parser.add_argument("--target", choices=available_datasets, default=None, help="Target")
     parser.add_argument('--comm_round', type = int, default=100, help = 'communication rounds')
     parser.add_argument('--wk_iters', type = int, default=1, help = 'optimization iters in local worker between communication')
     parser.add_argument('--mode', type = str, default='fedala', help='[FedAvg | FedProx | FedBN | FedAdam | FedNova | FedNovaBN | FedBABU | FedAda]')
     parser.add_argument('--save_path', type = str, default='D:/pythonProject/IOP-FL-main/io-pfl-exp/checkpoint/', help='path to save the checkpoint')
     parser.add_argument('--resume', action='store_true', help ='resume training from the save path checkpoint')
     parser.add_argument('--data', type = str, default='camelyon17', help='Different dataset: digits5, domainnet, office, pacs, brain')
     parser.add_argument('--gpu', type = str, default="0", help = 'gpu device number')
     parser.add_argument('--seed', type = int, default=0, help = 'random seed')
     parser.add_argument('--percent', type=float, default=1.0, help='percent of data used to train(1,0.75,0.5,0.25)')
     parser.add_argument('--client_optimizer', type = str, default='adam', help='local optimizer')
     parser.add_argument('--alpha', type = float, default=0.1, help='momentum weight for moving averaging')
     parser.add_argument('--test_time', type = str, default='mix', help='test time adaptation methods')
     parser.add_argument('--debug', action='store_true', help = 'use small data to debug')
     parser.add_argument('--test', action='store_true', help='test on local clients')
     parser.add_argument('--ood_test', action='store_true', help='test on ood client')
     parser.add_argument('--balance', action='store_true', help='do not truncate train data to same length')
     parser.add_argument('--every_save', action='store_true', help='Save ckpt with explicit name every iter')
             
     parser.add_argument('-s', "--rand_percent", type=int, default=80)
     parser.add_argument("--ala_times", type=int, default=10)
     parser.add_argument('-et', "--eta", type=float, default=1.0)
     parser.add_argument('-p', "--layer_idx", type=int, default=20,help="More fine-graind than its original paper.")
        
    # 添加 `lamda` 参数 pfedme
     parser.add_argument('--lr_lambda', type=float, default=0.01, help='Regularization parameter for pFedMe')
     parser.add_argument('--K', type=int, default=5, help='Number of local steps for pFedMe')
     parser.add_argument('--p_learning_rate', type=float, default=0.01, help='Personalized learning rate for pFedMe')
     parser.add_argument('--mu', type=float, default=0.001, help='Proximal regularization term coefficient for pFedMe')
     parser.add_argument('--local_epochs', type=int, default=5, help='Number of local epochs for each client')

     # 添加是否进行学习率衰减的参数
     parser.add_argument('--learning_rate_decay', action='store_true', help='Enable learning rate decay')
     parser.add_argument('--beta', type=float, default=0.5, help='Weight for beta aggregation in pFedMe (0.0 to 1.0)')
     # 添加参数 fedrep
     parser.add_argument('--lr_decay', type=float, default=0.99, help='The learning rate decay rate for exponential scheduler')
     parser.add_argument('--plocal_epochs', type=int, default=10, help='Number of local epochs for personalized training')
     # 添加 VAE 相关参数  
     parser.add_argument('--vae_latent_dim', type=int, default=16, help='Latent dimension for VAE')
     parser.add_argument('--vae_lr', type=float, default=0.005, help='Learning rate for VAE model')
     parser.add_argument('--vae_epochs', type=int, default=10, help='Number of epochs to train the VAE')
     parser.add_argument('--vae_train_epochs', type=int, default=5, help='Number of epochs for VAE training')
     # 添加参数 perfeavg
     parser.add_argument('--lambda_param', type=float, default=0.01, help='Regularization parameter for personalization')
    
     # 添加 APPLE 特有参数
     parser.add_argument('--L', type=float, default=0.5, help='Fraction of total global rounds for APPLE lambda scheduling')
     parser.add_argument('--mu_apple', type=float, default=0.001, help='Regularization coefficient for APPLE')
     parser.add_argument('--dr_learning_rate', type=float, default=0.001, help='Learning rate for dynamic weight adjustment in APPLE')
    
     parser.add_argument('--fds_lambda', type=float, default=0.1, help='Weight for FDS loss')
     parser.add_argument('--inner_lr_steps', type=int, default=1, help='Number of inner loop training steps')
     parser.add_argument('--ood_train_steps', type=int, default=1, help="Number of steps for OOD domain generalization training")
     parser.add_argument('--outer_lr_steps', type=int, default=1, help='Number of outer loop steps')  # 重要：添加此行
        
     args = parser.parse_args()
     # load exp default settings

     return args
     
