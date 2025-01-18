import torch
from torch.utils.data.dataloader import DataLoader

def generate_data_loader(args, client_num, trainsets, valsets, testsets, ood_set):
     # 创建字典来存储每个客户端的训练数据数量、本地训练数据、本地验证数据和本地测试数据
     train_data_local_num_dict = dict()
     train_data_local_dict = dict()
     val_data_local_dict = dict()
     test_data_local_dict = dict()
     # 初始化训练、验证和测试数据的总数
     train_data_num = 0
     val_data_num = 0
     test_data_num = 0
     # 获取所有训练集的最小长度
     min_data_len = min([len(s) for s in trainsets])

     # 如果设置了平衡数据集选项，则打印平衡数据集的信息
     if args.balance:
          print(f'Balance training set, using {args.percent*100}% training data')
     # 遍历每个训练集
     for idx in range(len(trainsets)):
          if args.balance:
               # 如果平衡数据集选项为真，则从每个训练集中提取最小长度百分比的数据
               trainset = torch.utils.data.Subset(trainsets[idx], list(range(int(min_data_len*args.percent))))
          else:
               # 否则，使用整个训练集
               trainset = trainsets[idx]
          valset = valsets[idx]
          testset = testsets[idx]
          # for debug
          # 如果调试模式为真，则只使用前1000个样本的数据子集
          if args.debug:
               trainset = torch.utils.data.Subset(trainsets[idx], list(range(1000)))
               valset = torch.utils.data.Subset(valsets[idx], list(range(1000)))
               testset = torch.utils.data.Subset(testsets[idx], list(range(1000)))

          # 打印 trainset 的一些基本信息
          print(f"[Client {idx}] Type of trainset: {type(trainset)}")
          print(f"[Client {idx}] Number of samples in trainset: {len(trainset)}")

          # 尝试打印 trainset 的第一个样本
          try:
              first_sample = trainset[0] 
              if isinstance(first_sample, tuple):
                  image, label = first_sample
                  print(f"[Client {idx}] Shape of the first training image: {image.shape}")
                  if hasattr(label, 'shape'):
                      print(f"[Client {idx}] Shape of the first training label: {label.shape}")
                  else:
                      print(f"[Client {idx}] Label type: {type(label)}, Label value: {label}")
              else:
                  print(f"[Client {idx}] First sample type: {type(first_sample)}")
          except Exception as e:
              print(f"[Client {idx}] Error while accessing the first sample: {e}")

          #print(f'[Client {sites[idx]}] Train={len(trainset)}, Val={len(valset)}, Test={len(testset)}')
          # 累加训练、验证和测试数据的数量
          train_data_num += len(trainset)
          val_data_num += len(valset)
          test_data_num += len(testset)
          # 记录每个客户端的训练数据数量
          train_data_local_num_dict[idx] = len(trainset)
          # 创建数据加载器并存储在相应的字典中
          train_data_local_dict[idx] = DataLoader(trainset, batch_size=args.batch, shuffle=True)
          val_data_local_dict[idx]   = DataLoader(valset, batch_size=args.batch, shuffle=False)
          test_data_local_dict[idx]  = DataLoader(testset, batch_size=4, shuffle=False)
     # 如果调试模式为真，则只使用OOD数据集的前1000个样本
     if args.debug:
          ood_set = torch.utils.data.Subset(ood_set, list(range(1000)))
     # 创建OOD数据加载器
     ood_loader = torch.utils.data.DataLoader(ood_set, batch_size=args.batch, shuffle=False)
     # print(len(ood_loader.dataset))
    
     sample_data, sample_label = ood_set[0]
     print(f"Sample data shape: {sample_data.shape}")
     print(f"Sample label: {sample_label}")
     # 返回客户端数量和所有数据集相关的信息
     return (client_num, [train_data_num, val_data_num, test_data_num, train_data_local_num_dict, train_data_local_dict, val_data_local_dict, test_data_local_dict, ood_loader])