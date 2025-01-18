from abc import ABC, abstractmethod


# 定义一个抽象基类 ModelTrainer，继承自 ABC（抽象基类）
class ModelTrainer(ABC):
    """Abstract base class for federated learning trainer.
       - This class can be used in both server and client side
       - This class is an operator which does not cache any states inside.
    """

    # 初始化方法，接收模型和参数作为输入
    def __init__(self, model, args=None):
        self.model = model  # 存储模型
        self.id = 0  # 初始化ID为0
        self.args = args  # 存储参数

    # 设置ID的方法
    def set_id(self, trainer_id):
        self.id = trainer_id  # 将输入的 trainer_id 赋值给实例变量 id

    # 抽象方法 get_model_params，子类必须实现
    @abstractmethod
    def get_model_params(self):
        pass

    # 抽象方法 set_model_params，子类必须实现
    @abstractmethod
    def set_model_params(self, model_parameters):
        pass

    # 注释掉的抽象方法 validate，子类可以选择实现
    # @abstractmethod
    # def validate(self, val_data, device, args=None):
    #     pass

    # 注释掉的抽象方法 test_on_the_server，子类可以选择实现
    # @abstractmethod
    # def test_on_the_server(self, train_data_local_dict, test_data_local_dict, device, args=None) -> bool:
    #     pass
