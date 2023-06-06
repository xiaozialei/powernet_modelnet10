import torch
import torch.optim as optim
import torch.nn.functional as F
from dataset import ShapeNetDataset
from model import PointNetDenseCls,feature_transform_regularizer


batchsize=16
feature_transform=True
# split可以为val
train_dataset = ShapeNetDataset(
    root='shapenetcore_partanno_segmentation_benchmark_v0',
    classification=False,
    class_choice='Airplane',
    split='train',
    data_augmentation=True)
# 一个样本是一个点云文件和对应的标签文件
train_dataloader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=batchsize,
    drop_last=True,
    shuffle=True,
    num_workers=0)

test_dataset = ShapeNetDataset(
    root='shapenetcore_partanno_segmentation_benchmark_v0',
    classification=False,
    class_choice='Airplane',
    split='test',
    data_augmentation=False)

testdataloader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=batchsize,
    drop_last=True,
    shuffle=True,
    num_workers=0)

print(len(train_dataset), len(test_dataset))
num_classes = train_dataset.num_seg_classes
print(num_classes)

model = PointNetDenseCls(k=num_classes,feature_transform=feature_transform)

optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
print(model)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)
criterion = torch.nn.NLLLoss().to(device)

epochs = 10
num_batch = len(train_dataset) / batchsize

for epoch in range(epochs):
    train_loss = 0.0
    train_accuracy =0.0
    model.train()
    
    for i,(pts,labels) in enumerate(train_dataloader):
        batch_correct = 0
        pts_num = 0
        
        labels = labels.to(device)
        pts =  pts.transpose(2,1)   
        pts= pts.to(device)

        optimizer.zero_grad()
        
        output,_,trans_feat = model(pts)
        # output(size,num_classes) labels(size)
        output = output.view(-1,num_classes)
        labels = labels.view(-1, 1)[:, 0]-1
        pts_num+=len(labels)
        loss = criterion(output,labels)
        if feature_transform:
            loss += feature_transform_regularizer(trans_feat) * 0.001
            
        pred = torch.argmax(output,dim=1)
        batch_correct += (pred==labels).sum().item()
        train_loss +=loss
        train_accuracy +=batch_correct/pts_num
        loss.backward()
        optimizer.step()
        
        print(f'cur batch is {i} total about 122')
    scheduler.step()    
    print(f'cur epoch {epoch},average train loss {train_loss/num_batch},accuracy rate {train_accuracy/num_batch}')
    