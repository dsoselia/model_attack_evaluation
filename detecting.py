# %%
import torch
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
from pathlib import Path
from resnet import ResNet18

# cifar10
train_transform = transforms.Compose(
    [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
    ]
)
test_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
    ]
)

train_set = torchvision.datasets.CIFAR10(
    root="./data", train=True, download=True, transform=train_transform
)
test_set = torchvision.datasets.CIFAR10(
    root="./data", train=False, download=True, transform=test_transform
)

# seperate train into two non-overlapping subsets


# train_1, train_2 = torch.utils.data.random_split(train_set, [25000, 25000])
# train_1, val_1 = torch.utils.data.random_split(train_1, [20000, 5000])
# stealing_set = torch.utils.data.Subset(train_1, range(5000))
# train_2, val_2 = torch.utils.data.random_split(train_2, [20000, 5000])

train_1, val_1 = train_set, test_set
train_2, val_2 = train_set, test_set
stealing_set = train_set

# create dataloaders
num_workers = 8
batch_size = 64


test_loader = torch.utils.data.DataLoader(
    test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers
)
train_loader_1 = torch.utils.data.DataLoader(
    train_1, batch_size=batch_size, shuffle=True, num_workers=num_workers
)
train_loader_2 = torch.utils.data.DataLoader(
    train_2, batch_size=batch_size, shuffle=True, num_workers=num_workers
)
val_loader_1 = torch.utils.data.DataLoader(
    val_1, batch_size=batch_size, shuffle=False, num_workers=num_workers
)
val_loader_2 = torch.utils.data.DataLoader(
    val_2, batch_size=batch_size, shuffle=False, num_workers=num_workers
)
stealing_loader = torch.utils.data.DataLoader(
    stealing_set, batch_size=batch_size, shuffle=False, num_workers=num_workers
)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# %%

# train a model on train_1 and validate on val_1

#
epochs = 50
optimizer_name = "sgd"
# model_1 = torchvision.models.resnet18()
# model_1.fc = torch.nn.Linear(512, 10)
model_1 = ResNet18()
model_1 = model_1.to(device)

# make model 2 deep copy of model 1
# model_2 = torchvision.models.resnet18()
# model_2.fc = torch.nn.Linear(512, 10)
model_2 = ResNet18()
model_2 = model_2.to(device)

# %%
def train_model(
    model,
    train_loader,
    val_loader,
    epochs=100,
    lr=0.002,
    momentum=0.9,
    optimizer_name="adam",
    save_path=None,
):
    print("Training model")
    print("_____Parameters_____")
    print(f"epochs: {epochs}")
    print(f"lr: {lr}")
    print(f"momentum: {momentum}")
    print(f"optimizer: {optimizer_name}")
    print("____________________")
    criterion = torch.nn.CrossEntropyLoss()
    if optimizer_name == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    elif optimizer_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    else:
        raise NotImplementedError("Optimizer not implemented")
    for epoch in range(epochs):
        model.train()
        print(f"Epoch {epoch}")
        running_loss = 0.0
        for i, data in tqdm(enumerate(train_loader, 0), total=len(train_loader)):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # validate
        correct = 0
        total = 0
        print("[%d, %5d] Train loss: %.3f" % (epoch, i + 1, running_loss / i))
        running_loss = 0.0
        with torch.no_grad():
            model.eval()
            for data in val_loader:
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print("Accuracy of the network on the val set: %d %%" % (100 * correct / total))
        if save_path is not None and epoch % 10 == 0:
            Path(save_path).mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), save_path + f"model_{epoch}.pth")
    print("Finished Training")

    return model


# train model_1
# %%
load_model = True

if load_model:
    model_1.load_state_dict(torch.load("models/model_1/model_20.pth"))
else:
    model_1 = train_model(
        model_1,
        train_loader_1,
        val_loader_1,
        epochs=epochs,
        save_path="./models/model_1/",
        optimizer_name=optimizer_name,
    )

# train model_2
if load_model:
    model_1.load_state_dict(torch.load("models/model_1/model_20.pth"))
else:
    model_2 = train_model(
        model_2,
        train_loader_2,
        val_loader_2,
        epochs=epochs,
        save_path="./models/model_2/",
        optimizer_name=optimizer_name,
    )


# %%
# train using stealing_set

# stolen_model = torchvision.models.resnet18(pretrained=False)
# stolen_model.fc = torch.nn.Linear(512, 10)
stolen_model = ResNet18()
stolen_model = stolen_model.to(device)

# %%
criterion = torch.nn.CrossEntropyLoss()
if optimizer_name == "sgd":
    optimizer = torch.optim.SGD(stolen_model.parameters(), lr=0.002, momentum=0.9)
elif optimizer_name == "adam":
    optimizer = torch.optim.Adam(stolen_model.parameters(), lr=0.001)


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
            print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0
    # evaluate acc on val_1 dataloader
    correct = 0
    total = 0
    with torch.no_grad():
        for data in val_loader_1:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = stolen_model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(
        "Accuracy of the Stolen model on the val set images: %d %%"
        % (100 * correct / total)
    )
#

# %%

# save stolen model
Path("models/stolen_model/").mkdir(parents=True, exist_ok=True)
torch.save(stolen_model.state_dict(), "models/stolen_model/stolen_model.pth")

# eval using paper

# Black-Box Setting: Blind Walk

# %%


def get_random_walk_diff_vector(
    model, x_0, y, step_size=0.1, random_direction_matrix=None
):
    with torch.no_grad():
        if len(x_0.shape) == 3:
            x_0 = x_0.unsqueeze(0)
        if random_direction_matrix is None:
            random_direction_matrix = torch.randn(x_0.shape)
            raise NotImplementedError("Not implemented for 3D tensors")

        x = x_0.clone().detach()
        y = y.clone().detach()
        x = x.to(device)
        y = y.to(device)
        random_direction_matrix = random_direction_matrix.to(device)
        model.eval()
        steps_taken = np.zeros(len(y))
        remaining = list(range(len(y)))
        while len(x) > 0 and steps_taken < 15:
            y_pred = model(x[remaining])
            new_remaining = (y_pred.max(1)[1] == y[remaining])
            remaining = remaining[new_remaining]
            x = x[remaining] + step_size * random_direction_matrix
            y = y[remaining]
            steps_taken[remaining] += 1

    return steps_taken


random_direction_matrix_list = []
random_directions_n = 8
for i in range(random_directions_n):
    random_direction_matrix_list.append(torch.randn(train_set[0][0].shape))


# %%

# iterate over train_1 and val_1 and call get_random_walk_diff_vector on each
def get_distances_for_model(model, dataset, random_direction_matrix_list, label):
    distances = []
    for inputs, labels in tqdm(dataset, total=len(dataset)):
        print("batch started")
        for random_direction_matrix in random_direction_matrix_list:
            steps_taken = get_random_walk_diff_vector(
                model, inputs, labels, random_direction_matrix=random_direction_matrix
            )
            distances += steps_taken
    return np.swapaxes(distances, 0, 1), [label for _ in range(len(distances))]


distances, distances_labels = get_distances_for_model(
    model_1, stealing_loader, random_direction_matrix_list, 0
)
# concatinate val_loader_1 distances
val_distances, val_distances_labels = get_distances_for_model(
    model_1, val_loader_1, random_direction_matrix_list, 1
)
distances = distances + val_distances
distances_labels = distances_labels + val_distances_labels
# %%

distances_labels[:50000] = [0] * 50000
distances_labels[50000:] = [1] * 10000
# %%

# save distances
distances_path = "distances"
Path(distances_path).mkdir(parents=True, exist_ok=True)
import pickle

with open(distances_path + "/distances.pkl", "wb") as f:
    pickle.dump(distances, f, pickle.HIGHEST_PROTOCOL)
with open(distances_path + "/distances_labels.pkl", "wb") as f:
    pickle.dump(distances_labels, f, pickle.HIGHEST_PROTOCOL)

# %%

# Load distances

with open(distances_path + "/distances.pkl", "rb") as f:
    distances = pickle.load(f)
with open(distances_path + "/distances_labels.pkl", "rb") as f:
    distances_labels = pickle.load(f)

# count how many reached max steps of 15

max_rached_count = 0
for distance in distances[:50000]:
    if max(distance) == 15:
        max_rached_count += 1
print(max_rached_count)

val_max_rached_count = 0
for distance in distances[50000:]:
    if max(distance) == 15:
        val_max_rached_count += 1
print(val_max_rached_count)

min_reached_count = 1
for distance in distances[:50000]:
    if min(distance) == 1:
        min_reached_count += 1
print(min_reached_count)

val_min_reached_count = 1
for distance in distances[50000:]:
    if min(distance) == 1:
        val_min_reached_count += 1
print(val_min_reached_count)


# %%
# balance by undersampling
import numpy as np

distance_classes = [0, 1]
samples_per_class = min([distances_labels.count(c) for c in distance_classes])

distances_balanced = []
distances_labels_balanced = []
for c in distance_classes:
    indices = [i for i, x in enumerate(distances_labels) if x == c]
    indices = np.random.choice(indices, samples_per_class, replace=False)
    for i in indices:
        distances_balanced.append(distances[i])
        distances_labels_balanced.append(distances_labels[i])

distances = distances_balanced
distances_labels = distances_labels_balanced

# %%
# train a  two-layer linear network distances and distances_labels binary classification


distances = torch.tensor(distances)
max_distance = torch.max(distances)
# distances = transform_distances(distances)
distances_labels = torch.tensor(distances_labels)
distances_labels = distances_labels.float()
distances = distances.float()
distances = distances / max_distance

# unsqueeze distance_labels
if len(distances_labels.shape) == 1:
    distances_labels = distances_labels.unsqueeze(1)
distances_dataset = torch.utils.data.TensorDataset(distances, distances_labels)
train_distances, val_distances = torch.utils.data.random_split(
    distances_dataset,
    [
        int(len(distances_dataset) * 0.8),
        len(distances_dataset) - int(len(distances_dataset) * 0.8),
    ],
)
train_distances_loader = torch.utils.data.DataLoader(
    train_distances, batch_size=4, shuffle=True, num_workers=num_workers
)
val_distances_loader = torch.utils.data.DataLoader(
    val_distances, batch_size=4, shuffle=False, num_workers=num_workers
)

# %%

# train a  two-layer linear network distances and distances_labels binary classification

#

# train a model on train_1 and validate on val_1


class BinaryModel(torch.nn.Module):
    def __init__(self):
        super(BinaryModel, self).__init__()
        self.fc1 = torch.nn.Linear(random_directions_n, 32)
        self.fc2 = torch.nn.Linear(32, 1)

    def forward(self, x):
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.sigmoid(self.fc2(x))
        return x


binary_classifier = BinaryModel()
binary_classifier = binary_classifier.to(device)

criterion_binary = torch.nn.BCELoss()

optimizer_binary = torch.optim.SGD(
    binary_classifier.parameters(), lr=0.001, momentum=0.9
)
binary_epochs = 20
for epoch in range(binary_epochs):

    running_loss = 0.0
    for i, data in enumerate(train_distances_loader):
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer_binary.zero_grad()

        outputs = binary_classifier(inputs)
        loss = criterion_binary(outputs, labels)
        loss.backward()
        optimizer_binary.step()

        running_loss += loss.item()
    print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1, running_loss / 2000))
    running_loss = 0.0

    correct = 0
    total = 0
    with torch.no_grad():
        for data in val_distances_loader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = binary_classifier(images)
            outputs = torch.round(outputs)
            total += labels.size(0)
            correct += (outputs == labels).sum().item()
        print(
            "Accuracy of the network on the val images: %d %%" % (100 * correct / total)
        )

print("Finished Training")
# %%

# random forests binary classification

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# scikit learn random split distances, distances_labels
from sklearn.model_selection import train_test_split

X_train_rf, X_test_rf, y_train_rf, y_test_rf = train_test_split(
    distances, distances_labels, test_size=0.2, random_state=42
)

rf = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)

rf.fit(X_train_rf, y_train_rf)

y_pred_rf = rf.predict(X_test_rf)

print(
    "Accuracy of the network on the val images: %d %%"
    % (100 * accuracy_score(y_test_rf, y_pred_rf))
)
# %%

# test model on dataset using binary classifier and return positive classification rate

stolen_distances, stolen_distances_labels = get_distances_for_model(
    stolen_model, train_loader_1, random_direction_matrix_list, 0
)
# %%
# get binary classification predictions on stolen_distances
stolen_distances = torch.tensor(stolen_distances)
stolen_distances = stolen_distances.float()
stolen_distances = stolen_distances / max_distance
# stolen_distances = transform_distances(stolen_distances)
stolen_distances_labels = torch.tensor(stolen_distances_labels)


stolen_distances_dataset = torch.utils.data.TensorDataset(
    stolen_distances, stolen_distances_labels
)

stolen_distances_loader = torch.utils.data.DataLoader(
    stolen_distances_dataset, batch_size=4, shuffle=False, num_workers=num_workers
)

# %%
positive_predictions = 0
with torch.no_grad():
    for data in stolen_distances_loader:
        distances, labels = data
        distances = distances.to(device)
        labels = labels.to(device)
        outputs = binary_classifier(distances)
        outputs = torch.round(outputs)
        # as int
        outputs = outputs.int()
        # count number of 1s
        positive_predictions += torch.sum(outputs).item()
positive_percentage = positive_predictions / len(stolen_distances_loader.dataset)

print("positive predictions percentage: ", positive_percentage)
# %%


def test_if_stolen(
    model, dataloader, binary_classifier, random_direction_matrix_list, max_distance
):
    stolen_distances, stolen_distances_labels = get_distances_for_model(
        model, dataloader, random_direction_matrix_list, 0
    )
    stolen_distances = torch.tensor(stolen_distances)
    stolen_distances = stolen_distances.float()
    stolen_distances = stolen_distances / max_distance
    stolen_distances_labels = torch.tensor(stolen_distances_labels)
    stolen_distances_dataset = torch.utils.data.TensorDataset(
        stolen_distances, stolen_distances_labels
    )
    stolen_distances_loader = torch.utils.data.DataLoader(
        stolen_distances_dataset, batch_size=4, shuffle=False, num_workers=num_workers
    )
    positive_predictions = 0
    with torch.no_grad():
        for data in stolen_distances_loader:
            distances, labels = data
            distances = distances.to(device)
            labels = labels.to(device)
            outputs = binary_classifier(distances)
            outputs = torch.round(outputs)
            # as int
            outputs = outputs.int()
            # count number of 1s
            positive_predictions += torch.sum(outputs).item()
    positive_percentage = positive_predictions / len(stolen_distances_loader.dataset)
    return positive_percentage


# call on stolen_model and train_1 loader

stolen_model_score = test_if_stolen(
    stolen_model,
    stealing_loader,
    binary_classifier,
    random_direction_matrix_list,
    max_distance,
)


model_2_score = test_if_stolen(
    model_2,
    stealing_loader,
    binary_classifier,
    random_direction_matrix_list,
    max_distance,
)

print("stolen model score: ", stolen_model_score)

print("model_2 score: ", model_2_score)
# %%
# %%
