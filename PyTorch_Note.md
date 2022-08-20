[![hackmd-github-sync-badge](https://hackmd.io/2YtqI8YBTVunvXfq9LIWGw/badge)](https://hackmd.io/2YtqI8YBTVunvXfq9LIWGw)
###### tags: `Pytorch self-study`
PyTorch筆記
===
:::spoiler 筆記目錄
[TOC]
:::

基本步驟
---
:::success
    
1. [定義模型結構](#建立神經網路)
2. [定義dataset、dataLoader](#DataSet-and-DataLoader)
3. [定義超參數: Batch, Epoch, Learning](#最佳化模型參數)
4. [選擇或自己定義Loss function](#損失函數)
5. [選擇或定義optimizer](#Optimizer)
6. [將訓練過程包裝成函式，inference也包裝成函式](#訓練過程實作)
7. [將模型、dataset、dataLoader等等實體化(將訓練資料作為參數傳進去dataset)](#執行訓練)
8. [呼叫訓練函式，並將需要的各種參數帶入](#執行訓練)
9. [儲存模型以及權重](#儲存及載入模型)

:::
Tensor
---

* 多維度矩陣與numpy相似
* 可以在GPU上進行運算，numpy不能
* numpy array 可以轉換成tensor
* 預設數值型態 float32

### 初始化Tensor
#### numpy to tensor四種方法如下: 

:::info
* torch.tensor()
* torch.Tensor()
* torch.as_tensor()
* torch.from_numpy()
:::



使用的數值型態:
float32 -> Tensor()
int64 -> tensor()、as_tensor()、from_numpy()


使用shallow copy: 
as_tensor()、from_numpy()
若不改變資料內容，shallow copy比較有效率

使用deep copy: 
Tensor()、tensor()

##### 直接宣告: 
```python=
data = [[1, 2],[3, 4]]
x_data = torch.tensor(data)
```



##### 其他方法: 
```python=
# 參數shape為tuple型態
shape = (2,3,) 
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)
```


### Tensor的屬性
    tensor.shape
    tensor.dtype
    tensor.device 該tensor儲存在cpu or gpu

預設tensor初始化建立在cpu上，如果可使用gpu則可以用<span style="color: red">to()</span>函式，將tensor搬移到gpu，但搬移龐大資料也會使用大量資源

    if torch.cuda.is_available():
        tensor = tensor.to("cuda")

### Tensor Indexing
tensor[row][col]
先列在行

### Tensor 運算及符號
:::info
* @ : 矩陣相乘
* tensor.T : 轉置矩陣
* tensor1.matmul(tensor2) === tensor1 @ tensor2
* tensor.sum() 會回傳只有一個元素的tensor
* item()函式，將單一元素tensor轉換成python數值基本型態
:::

    
#### In-place Operation
具有 "_" 後綴的函式，不建議使用

    例如: tensor.add_(5) 會直接影響到tensor的數值
:::warning
In-place operations 可以節省記憶體，但是可能會在計算微分時產生問題，因此不建議使用
:::   


DataSet and DataLoader
---
PyTorch 提供兩個處理資料的工具

    torch.utils.data.DataLoader
    torch.utils.data.Dataset
    
Dataset 儲存資料的樣本以及標籤
DataLoader 用於包裝Dataset，方便從外部迭代使用Dataset的資料


### 客製化Dataset

```python=
# 建立class包含三個函式，__init__, __len__, and __getitem__
import torch
from torch.utils.data import Dataset
import os
import pandas as pd
from torchvision.io import read_image
import torchvision.transforms as transforms

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
    
#main
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
train_dataset = Dataset('xxx.csv',img_dir_train, transforms)
test_dataset = Dataset('xxx.csv',img_dir_test, transforms)  
```


    __init__: 當物件被建立時，會用來初始化目標資料，建立資料結構，例如: directory、dataframe
    
    __len__: 回傳資料的總長度
    
    __getitem__: 可以接收索引值idx並回傳dataset中該索引值代表的資料，舉例來說，該索引值的資料可以是
    圖片在硬碟上的位址以及標籤，讀取該圖片以及標籤並做客製化的轉變(轉成tensor之類的)，再回傳
    

### 使用DataLoader幫助訓練時讀取資料
有時，為了防止過度擬合，會希望每次從dataset中取出的資料可以是隨機抽取的，或是使用特定的batch size訓練資料，又或者是希望可以使用Python的平行處理加速運算，
因此DataLoader在這時就擔任了包裝上述功能的工具。

```python=
from torch.utils.data import DataLoader

train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=True)
```

### 使用DataLoader迭代資料
    next()
    
```python=
train_features, train_labels = next(iter(train_dataloader))
#根據先前定義的batch size train_feature, train_label 一次會取出一排資料
print(f"Feature batch shape: {train_features.size()}") #torch.Size([64, 1, 28, 28])
print(f"Labels batch shape: {train_labels.size()}") #torch.Size([64])
img = train_features[0].squeeze() #將1*32*32 -> 32*32方便下方imshow使用
label = train_labels[0] #ground truth
plt.imshow(img, cmap="gray")  #img 參數維度規定看下方連結
plt.show()
print(f"Label: {label}")
```
[matplotlib.pyplot.imshow](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.imshow.html)

建立神經網路
---
### 定義Network class 以LeNet-5為例，辨識0~9手寫數字
![](https://i.imgur.com/TOj4htP.png)

* C1卷積層，6 filters(5*5)、stride=1
* S2池化層，2*2 stride=2
* C3卷積層，16 filters(5*5)、stride=1
* S4池化層S4，2*2 stride=2
* F5全連接層
* F6全連接層

[CNN卷積神經網路](/GUk1nQxTTMq-nmUwZ7ZcRw)


基於 PyTorch 提供的nn.Module建立class

```python=

import torch                     # for all things PyTorch
import torch.nn as nn            # for torch.nn.Module, the parent object for PyTorch models
import torch.nn.functional as F  # for the activation function

#繼承來自nn.Module內容
class LeNet(nn.Module):

    def __init__(self):
        super(LeNet, self).__init__()
        #input 為1 channel, output為6 channel, kernel size=3
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.fc1 = nn.Linear(16 * 5 * 5, 120) 
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension. 
        '''Ex. when batch=16, size: [16, 1, 32, 32]
        size[1:]=[1, 32, 32]'''
        
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
    
```


### 使用已建立的神經網路
將建立出來的物件搬移至目標裝置(cpu or gpu)(模型預設使用cpu)
將input data傳遞進神經網路，但不要直接使用model.forward()
範例如下:

```python=
#main
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

model = LeNet().to(device)
input_1 = torch.rand(16,1,32,32) # batch, channel, 32*32shape of image
input_1 = input.to(device)
output = model(input_1)
print(output.shape) #torch.size([16, 10])
```
為了加快運算速度，建立好模型後將模型以及輸入轉移至gpu，再呼叫模型執行運算。最後，輸出的維度是batch size, output dimention，也就是建立模型時nn.Linear(84, 10)的10。
換句話說，我們建立的input data是以16張32*32單通道的黑白影像為輸入，最後這16張各自會有10 個output，分別是每一張照片可能是0~9的機率值。
:::danger
注意這邊做inference之前沒有轉換model.eval()，是因為LetNet並沒有batch_normalization layer或是dropout layer

[參考網站](https://stackoverflow.com/questions/60018578/what-does-model-eval-do-in-pytorch)
:::

### 解釋範例Model Layers

    nn.Flatten() 降維
    nn.Linear() 根據權重與偏差對輸入執行線性轉換，全連接層
    nn.ReLU() 可以創造非線性的轉換，增加模型自由度
    nn.Sequential() 有序的Layer Container，輸入的資料會按照函式中定義的排列順序傳遞
    nn.Softmax() 從模型輸出的值可能介於正負無限大，我們希望輸出可以控制在一定範圍
    
    
### 模型參數
nn.Module會自動記錄所有模型中的參數，只要使用parameters()、named_parameters()函式就可以取得各個參數

```python=
for name, param in model.named_parameters():
    print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")
```

:::spoiler output
    Layer: linear_relu_stack.0.weight | Size: torch.Size([512, 784]) | Values : tensor([[-0.0201,  0.0343,  0.0235,  ...,  0.0136,  0.0200, -0.0199],
        [-0.0053, -0.0221,  0.0116,  ..., -0.0276, -0.0324, -0.0082]],
       device='cuda:0', grad_fn=<SliceBackward0>)

    Layer: linear_relu_stack.0.bias | Size: torch.Size([512]) | Values : tensor([-0.0172,  0.0337], device='cuda:0', grad_fn=<SliceBackward0>)

    Layer: linear_relu_stack.2.weight | Size: torch.Size([512, 512]) | Values : tensor([[-0.0178, -0.0427,  0.0070,  ...,  0.0317, -0.0242, -0.0343],
            [-0.0067, -0.0245, -0.0051,  ..., -0.0015, -0.0149,  0.0034]],
           device='cuda:0', grad_fn=<SliceBackward0>)

    Layer: linear_relu_stack.2.bias | Size: torch.Size([512]) | Values : tensor([-0.0365, -0.0348], device='cuda:0', grad_fn=<SliceBackward0>)

    Layer: linear_relu_stack.4.weight | Size: torch.Size([10, 512]) | Values : tensor([[ 0.0301,  0.0097,  0.0155,  ...,  0.0222,  0.0327, -0.0281],
            [-0.0241,  0.0183, -0.0100,  ..., -0.0068,  0.0257,  0.0255]],
           device='cuda:0', grad_fn=<SliceBackward0>)

    Layer: linear_relu_stack.4.bias | Size: torch.Size([10]) | Values : tensor([-0.0323,  0.0104], device='cuda:0', grad_fn=<SliceBackward0>)
:::


torch.autograd()自動計算微分引擎
---
訓練神經網路最常使用的演算法back propagation，參數會隨著損失函數的梯度調整。
PyTorch內建自動計算微分的工具torch.autograd()



```python=
import torch

x = torch.ones(5)  # input tensor
y = torch.zeros(3)  # expected output
w = torch.randn(5, 3, requires_grad=True)
b = torch.randn(3, requires_grad=True)
z = torch.matmul(x, w)+b
loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)
```


    
==因為後面w、b需要取得grad，需要將requires_grad設為True==

![](https://i.imgur.com/ulJqRyw.png)

### 計算微分

為了最佳化參數w、b，需要知道LOSS對w以及b的偏微分。
呼叫loss.backward()來計算篇微分，
再使用w.grad、b.grad取得計算結果。


    loss.backward()
    print(w.grad)
    print(b.grad)

使用backward進行梯度的計算只能使用一次，如果需要進行多次同樣的計算要將參數retain_graph=True傳入backward()

### 停止紀錄梯度

所有requires_grad的tensor會自動被記錄計算的過程來求得梯度；有時候只是單純的希望輸入資料可以經由已定的參數計算出一個結果，並沒有要更新參數本身，這時候就需要停止記錄梯度。使用<span style="color: red">torch.no_grad()</span>，可以加快計算速度增加效率。



```python=
z = torch.matmul(x, w)+b
print(z.requires_grad)

with torch.no_grad():
    z = torch.matmul(x, w)+b
print(z.requires_grad)
```
    
或是<span style="color: red">tensor.detach()</span>

```python=
z = torch.matmul(x, w)+b
z_det = z.detach()
print(z_det.requires_grad)
```
    
#### 其他希望停止追蹤梯度的狀況
:::info
* 將部分網路中的參數設為frozen parameter，[finetuing預訓練的模型](https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html)
:::
### 其他關於計算微分的細節
autograd利用DAG有向無環圖記錄tensor以及所有執行的運算，在DAG中包含許多Function物件，葉節點是input tenors，根節點是output tensor。

執行forward pass時，autograd利用已知的參數計算結果與維持DAG中的梯度函式

當backward()在DAG根節點被呼叫，會執行backward pass，這時autograd會自動計算梯度、將計算完的結果累積存在.grad屬性中，使用chain rule將計算過程延伸直到葉節點。

### 將累積的梯度重置
:::warning
若是需要多次使用backward()，再每一次使用前呼叫tensor.grad.zero重置累積的梯度內容，才能得到正確的結果。
:::


最佳化模型參數
---
### 設定超參數

    Batch: 決定一次有多少資料要通過模型，通過一次後就更新一次所有參數
    Epoch: 要對Dataset進行多少次迭代，每次迭代會將整個dataset訓練過一次，嘗試使結果收斂，
            迭代每次結束前會有驗證或是測試資料進行確認，查看訓練成果
    Learning Rate: 更新參數時要一次更新的多寡，太大會導致訓練的不穩定，太小會導致訓練速度緩慢
    
### 損失函數
用於量測預測值與真實值的相似程度，可以自由定義差異
回歸類型的任務可以使用nn.MSELoss、分類型任務可以用nn.NLLLoss(Nagative Log)。
[各種Loss Function](https://pytorch.org/docs/stable/nn.html#loss-functions)

    loss_fn = nn.MSELoss()

### Optimizer
為了降低損失loss需要調整參數，調整的步驟就是Optimizer的工作。
調整的演算法可以自己選擇，使用[torch.optim](https://pytorch.org/docs/stable/optim.html)提供的optimizer，並且將learning rate作為參數傳進optimizer。

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    
在訓練過程中的最佳化包含三個步驟
:::info
1. 呼叫 optimizer.zero_grad() 重置模型的參數的梯度
2. 呼叫loss.backward()，PyTorch會自動將每個參數的梯度儲存
3. 計算出梯度之後，呼叫optimizer.step()根據梯度調整參數
:::


### 訓練過程實作
將訓練過程包裝成函式
[model.train詳細內容](###儲存模型的權重參數)

```python=
#省略建立模型過程

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train() #模型預設為訓練狀態，只是保險一點的寫法
    for batch, (X, y) in enumerate(dataloader):
        # 將輸入傳進模型並得到預測值
        pred = model(X)
        #計算預測值與真實值的差異
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()    #歸零
        loss.backward()    #計算梯度
        optimizer.step()    #根據梯度更新參數

        if batch % 100 == 0: #每通過batch個數量的資料就輸出一次損失的值
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    #size為所有資料的總長度
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0
    model.eval() #轉換成inference狀態
    #因為只是測試並不是要更新參數，所以使用no_grad使其停止追蹤梯度
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            #將輸出限制在範圍內，紀錄一個batch中有幾個正確的結果
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    #計算所有資料中有正確的比率
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
```
### 執行訓練
    
```python=
#舉例
batch=64
epochs = 10
learning_rate=1e-3
    
model = LeNet().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)
print("Done!")
```

儲存及載入模型
---
```python=
import torch
import torchvision.models as models
```
### 儲存模型的權重參數
    
```python=
#PyTorch將訓練的參數存在internal state dictionary
#以vgg16為例
#儲存
model = models.vgg16(pretrained=True)
torch.save(model.state_dict(), 'model_weights.pth')

#載入
model = models.vgg16(pretrained=false) #先建立與目標權重的結構相同的模型實體才能載入權重
model.load_state_dict(torch.load('model_weights.pth'))
model.eval()
```


:::warning
**pretrained=True** 會載入預設的參數，如果要載入自己的權重檔，可以將其設為false
:::
:::danger
**model.eval()** 有些模型中的結構(拋棄層、batchnorm)在訓練以及最後測試(inference、evaluate)的時候具有不同的表現，因此需要eval()來改變模型的表現；若要再回到訓練狀態則要model.train()

[參考網站](https://stackoverflow.com/questions/60018578/what-does-model-eval-do-in-pytorch)
:::

### 儲存模型的結構
```python=
torch.save(model, 'model.pth')
model = torch.load('model.pth')
```
    

