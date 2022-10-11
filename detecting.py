# %%
import torch
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm

# cifar10
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)

train_set = torchvision.datasets.CIFAR10(
    root="./data", train=True, download=True, transform=transform
)
test_set = torchvision.datasets.CIFAR10(
    root="./data", train=False, download=True, transform=transform
)

# seperate train into two non-overlapping subsets


train_1, train_2 = torch.utils.data.random_split(train_set, [25000, 25000])
train_1, val_1 = torch.utils.data.random_split(train_1, [20000, 5000])
stealing_set = torch.utils.data.Subset(train_1, range(5000))
train_2, val_2 = torch.utils.data.random_split(train_2, [20000, 5000])

# create dataloaders
num_workers = 8
batch_size = 128


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
epochs = 10
model_1 = torchvision.models.resnet18()
model_1.fc = torch.nn.Linear(512, 10)
model_1 = model_1.to(device)

# make model 2 deep copy of model 1
model_2 = torchvision.models.resnet18()
model_2.fc = torch.nn.Linear(512, 10)
model_2 = model_2.to(device)

# %%
def train_model(model, train_loader, val_loader, epochs=100, lr=0.01, momentum=0.9):
    print("Training model")
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    for epoch in range(epochs):
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
            if i % 200 == 0:
                print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
        # validate
        correct = 0
        total = 0
        with torch.no_grad():
            for data in val_loader:
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(
            "Accuracy of the network on the 10000 test images: %d %%"
            % (100 * correct / total)
        )
    return model


# train model_1

model_1 = train_model(model_1, train_loader_1, val_loader_1, epochs=epochs)

# train model_2

model_2 = train_model(model_2, train_loader_2, val_loader_2, epochs=epochs)


# %%
# train using stealing_set

stolen_model = torchvision.models.resnet18(pretrained=False)
stolen_model.fc = torch.nn.Linear(512, 10)
stolen_model = stolen_model.to(device)

# %%
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
        "Accuracy of the Stolen model on the 10000 test images: %d %%"
        % (100 * correct / total)
    )
#

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
            random_direction_matrix = torch.randn(x.shape)
            raise NotImplementedError("Not implemented for 3D tensors")

        x = x_0.clone().detach()
        y = y.clone().detach()
        x = x.to(device)
        y = y.to(device)
        random_direction_matrix = random_direction_matrix.to(device)
        y_pred = y
        model.eval()
        steps_taken = 0
        while y_pred == y and steps_taken < 5:
            y_pred = model(x)
            y_pred = torch.argmax(y_pred, dim=1)
            x = x + step_size * random_direction_matrix
            steps_taken += 1

    return steps_taken


random_direction_matrix_list = []
random_directions_n = 2
for i in range(random_directions_n):
    random_direction_matrix_list.append(torch.randn(train_set[0][0].shape))

# %%

# iterate over train_1 and val_1 and call get_random_walk_diff_vector on each
def get_distances_for_model(model, dataset, random_direction_matrix_list, label):

    distances = []
    distances_labels = []
    for inputs, labels in tqdm(dataset, total=len(dataset)):
        print("batch started")
        for img, label in zip(inputs, labels):
            k = []
            for random_direction_matrix in random_direction_matrix_list:
                steps_taken = get_random_walk_diff_vector(
                    model, img, label, random_direction_matrix=random_direction_matrix
                )
                k.append(steps_taken)
            distances.append(k)
            distances_labels.append(label)
    return distances, distances_labels


distances, distances_labels = get_distances_for_model(
    model_1, train_loader_1, random_direction_matrix_list, 0
)
# concatinate val_loader_1 distances
val_distances, val_distances_labels = get_distances_for_model(
    model_1, val_loader_1, random_direction_matrix_list, 1
)
distances = distances + val_distances
distances_labels = distances_labels + val_distances_labels

# %%

# train a  two-layer linear network distances and distances_labels binary classification
# transform_distances = transforms.Compose(
#     [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
# )

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
binary_epochs = 5
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
        if i % 2000 == 1999:
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
    train_loader_1,
    binary_classifier,
    random_direction_matrix_list,
    max_distance,
)


model_2_score = test_if_stolen(
    model_2,
    train_loader_1,
    binary_classifier,
    random_direction_matrix_list,
    max_distance,
)

print("stolen model score: ", stolen_model_score)

print("model_2 score: ", model_2_score)
# %%
