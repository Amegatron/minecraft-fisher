import torch
from dataset import FishermanSimplifiedDataset
from model import FishermanModel
from model_color import FishermanModelColor
from util import get_classes, train_model, get_state_filename
import torch.utils.data as data
import sys

if len(sys.argv) < 2:
    print("Trains the bot using dataset prepared by prepare.py\n")
    print("Usage:\n")
    print("    python train.py <dataset_path> [<save_filename>] [<learning_rate>] [<train_epochs>] [<target_acc>]\n")
    print("    <dataset_path>  - path with the dataset made with prepare.py")
    print("    <save_filename> - where to save the trained model")
    print("    <learning_rate> - learning rate (advanced)")
    print("    <train_epochs>  - how many epochs to train (advanced)")
    print("    <target_acc>    - desired target accuracy of a model (advanced)")
    exit(1)

if torch.cuda.is_available():
    device = torch.cuda.current_device()
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    print("CUDA is available. Using device \'%s\'" % torch.cuda.get_device_name(device))
else:
    device = torch.device('cpu')
    print("CUDA is NOT available, using CPU (training will be slow)")

dataset_path = sys.argv[1]

trained_model_path = get_state_filename()
if len(sys.argv) > 2:
    trained_model_path = sys.argv[2]

lr = 0.000012
if len(sys.argv) > 3:
    lr = float(sys.argv[3])

epochs = 500
if len(sys.argv) > 4:
    epochs = int(sys.argv[4])

target_acc = 0.97
if len(sys.argv) > 5:
    target_acc = float(sys.argv[5])

classes = get_classes()

print("Loading dataset...")
dataset = FishermanSimplifiedDataset(dataset_path, classes)
train_len = int(len(dataset) * 0.9)
train_dataset, test_dataset = data.random_split(dataset, (train_len, len(dataset) - train_len), generator=torch.Generator(device))
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, generator=torch.Generator(device))
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=True, generator=torch.Generator(device))

model = FishermanModelColor(len(classes)).to(device)

print("Training...")
train_model(
    model,
    epochs,
    torch.nn.CrossEntropyLoss(),
    torch.optim.Adam(model.parameters(), lr),
    train_dataloader,
    test_dataloader,
    device,
    target_acc=target_acc,
    verbose=True,
)

print("Saving model...")
torch.save(model.state_dict(), trained_model_path)
print("Saved to %s" % trained_model_path)
