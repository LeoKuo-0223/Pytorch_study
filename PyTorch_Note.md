###### tags: `Pytorch self-study`
Pytorch
===


Tensor
---

* 多維度矩陣與numpy相似
* 可以在GPU上進行運算，numpy不能
* numpy array 可以轉換成tensor
* 預設數值型態 float32

### 初始化Tensor
##### numpy to tensor四種方法如下: 


    torch.tensor()
    torch.Tensor()
    torch.as_tensor()
    torch.from_numpy()


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
    @ : 矩陣相乘
    tensor.T : 轉置矩陣
    tensor1.matmul(tensor2) === tensor1 @ tensor2
    
    tensor.sum() 會回傳只有一個元素的tensor
    使用item()函式，將單一元素tensor轉換成python數值基本型態
    
#### In-place Operation
具有 "_" 後綴的函式，不建議使用

    例如: tensor.add_(5) 會直接影響到tensor的數值
    
>     In-place operations save some memory, 
>     but can be problematic when computing derivatives because of an imme-
>     diate loss of history. 
>     Hence, their use is discouraged.

---

DataSet and DataLoader
---
PyTorch 提供兩個處理資料的工具

    torch.utils.data.DataLoader
    torch.utils.data.Dataset
    
Dataset 儲存資料的樣本以及標籤
DataLoader 用於包裝Dataset，方便從外部迭代使用Dataset的資料


#### 客製化Dataset

```python=
# 建立class包含三個函式，__init__, __len__, and __getitem__
import torch
from torch.utils.data import Dataset
import os
import pandas as pd
from torchvision.io import read_image

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
```


    __init__:
    當物件被建立時，會用來初始化目標資料，建立資料結構，例如: directory、dataframe
    
    __len__: 
    回傳資料的總長度
    
    __getitem__:
    可以接收索引值idx並回傳dataset中該索引值代表的資料，舉例來說，該索引值的資料可以是
    圖片在硬碟上的位址以及標籤，讀取該圖片以及標籤並做客製化的轉變(轉成tensor之類的)，再回傳
    

#### 使用DataLoader幫助訓練時讀取資料
有時，為了防止過度擬合，會希望每次從dataset中取出的資料可以是隨機抽取的，或是使用特定的batch size訓練資料，又或者是希望可以使用Python的平行處理加速運算，
因此DataLoader在這時就擔任了包裝上述功能的工具。

```python=
from torch.utils.data import DataLoader

train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)
```

##### 使用DataLoader迭代資料
    next()
    
```python=
train_features, train_labels = next(iter(train_dataloader))
#根據先前定義的batch size train_feature, train_label 一次會取出一排資料
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")
img = train_features[0].squeeze()
label = train_labels[0]
plt.imshow(img, cmap="gray")
plt.show()
print(f"Label: {label}")
```









