# %%
import torch
import torchvision
import torchvision.transforms as transforms


# cifar10

train_set = torchvision.datasets.CIFAR10()
test_set = torchvision.datasets.CIFAR10(train=False)

# seperate train into two non-overlapping subsets

train_1, train_2 = torch.utils.data.random_split(train_set, [25000, 25000])

train_1, val_1 = torch.utils.data.random_split(train_1, [20000, 5000])
stealing_set = torch.utils.data.Subset(train_1, range(5000))
train_2, val_2 = torch.utils.data.random_split(train_2, [20000, 5000])

# create dataloaders
num_workers = 2
batch_size = 32


test_loader = torch.utils.data.DataLoader(test_set, batch_size=4, shuffle=False, num_workers=num_workers)
train_loader_1 = torch.utils.data.DataLoader(train_1, batch_size=4, shuffle=True, num_workers=num_workers)
train_loader_2 = torch.utils.data.DataLoader(train_2, batch_size=4, shuffle=True, num_workers=num_workers)
val_loader_1 = torch.utils.data.DataLoader(val_1, batch_size=4, shuffle=False, num_workers=num_workers)
val_loader_2 = torch.utils.data.DataLoader(val_2, batch_size=4, shuffle=False, num_workers=num_workers)
stealing_loader = torch.utils.data.DataLoader(stealing_set, batch_size=4, shuffle=False, num_workers=num_workers)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# %%

# train a model on train_1 and validate on val_1

# 
epochs = 100
model_1 = torchvision.models.resnet18(pretrained=False)
model_1.fc = torch.nn.Linear(512, 10)
model_1 = model_1.to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model_1.parameters(), lr=0.001, momentum=0.9)

for epoch in range(epochs):
    running_loss = 0.0
    for i, data in enumerate(train_loader_1, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        outputs = model_1(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        if i % 2000 == 1999:
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

# 

# train using stealing_set

stolen_model = torchvision.models.resnet18(pretrained=False)
stolen_model.fc = torch.nn.Linear(512, 10)
stolen_model = stolen_model.to(device)

criterion = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(stolen_model.parameters(), lr=0.001, momentum=0.9)

for epoch in range(epochs):
    running_loss = 0.0
    for i, data in enumerate(stealing_loader, 0):
        inputs, _ = data
        # obtain labels from model_1
        inputs = inputs.to(device)
        labels = model_1(inputs)
        labels = torch.argmax(labels, dim=1)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        outputs = stolen_model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        if i % 2000 == 1999:
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

#

# eval using paper