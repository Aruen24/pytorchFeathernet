1. trian_3D.py
训练浮点型模型

2. convert_to_onnx.py
在量化之前，需要先将pytorch训练的模型，转换为onnx模型。

3. quant_model_and_test.py
普通定点化操作
add_normal(): 归一化运算add_pre_norm,层合并操作merge_layers等。
quant_moedel(): 普通定点化操作ndk.quantize.quantize_model

4. ./quant_train_torch/buildnet.py
在量化训练之前，运行得到网络结构的每一层。

5. train_3D_quant.py
量化训练

注:
class get_3d_liveness_data(data.Dataset):
    def __init__(self,data_path,flag):
        self.data=self.make_data(data_path)
        if flag=='test':
            self.trans = trans.Compose([
                trans.Grayscale(num_output_channels=1),
                trans.ToTensor(),
                #trans.Normalize([0.5], [0.5]),
            ])
        else:
            self.trans=trans.Compose([
                trans.Grayscale(num_output_channels=1),
                trans.RandomResizedCrop((224, 224), scale=(0.92, 1), ratio=(1, 1)),
                trans.RandomHorizontalFlip(),
                trans.ColorJitter(brightness=0.5),
                trans.ToTensor(),
                #trans.Normalize([0.5], [0.5]),
            ])

在用浮点型模型时，需要加上归一化#trans.Normalize([0.5], [0.5])。
在用ndk量化的模型，不需要。

