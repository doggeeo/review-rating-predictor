import torch
from torch.utils.data import DataLoader,TensorDataset
import pandas as pd

def save_checkpoint(model, optimizer, filename="my_checkpoint.pth"):
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)

def load_checkpoint(checkpoint_file, model, optimizer, lr):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location="cuda")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
device="cuda" if torch.cuda.is_available() else "cpu"
print(device)
classifier=torch.nn.Sequential(
    torch.nn.Linear(in_features=768,out_features=5),
    torch.nn.Softmax(dim=1)
).to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001, weight_decay=0.0001)
save_model = True
load_model = True
if load_model:
    load_checkpoint("classifier.pth", classifier, optimizer, 0.001,) #D://PycharmProjects//MLP//classifier.pth


data=pd.read_csv('dataset.csv',encoding='utf-8',sep=';')
num_epochs = 1000
batch_size=10
data=TensorDataset(torch.tensor(data.iloc[:,1:].to_numpy(dtype="float32")),torch.tensor(data['rating'].to_numpy())-1)
loader=DataLoader(dataset=data,batch_size=batch_size,shuffle=True)
for epoch in range(num_epochs):
    for i, (text, rating) in enumerate(loader):
        text=text.to(device)
        rating=rating.to(device)
        rating_pred = classifier(text).to(device)
        loss = criterion(rating_pred, rating).to(device)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f'Epoch [{epoch + 1}/{i}], Loss: {loss.item():.4f}')
    save_checkpoint(classifier, optimizer, filename="classifier.pth")
