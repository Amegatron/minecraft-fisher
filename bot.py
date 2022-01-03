from util import find_window, get_transform, get_classes, get_state_filename
from model import FishermanModel
from fisherman import Fisherman
from time import sleep
import sys
import torch

if len(sys.argv) < 2:
    print("Usage:\n")
    print("    python bot.py <minecraft_window_title> [<save_filename>]\n")
    print("Where <minecraft_window_title> is a the title (may be partial) of Minecraft,\nfor example Minecraft 1.18")
    print("And optional <save_filename> is the file made by train.py\n")
    exit(1)

minecraft_title = sys.argv[1]
model_path = get_state_filename()
if len(sys.argv) > 2:
    model_path = sys.argv[2]

# Find Minecraft window
windows = find_window(minecraft_title)
if len(windows) == 0:
    print("Could not find minecraft window")
    exit(1)

# Load model
print("Loading model...")
classes = get_classes()
model = FishermanModel(len(classes))
model.load_state_dict(torch.load(model_path))
model.eval()

hwnd = windows[0][0]
fishermanBot = Fisherman(
    hwnd,
    model,
    [
        Fisherman.STATE_FISHING_IDLE,
        Fisherman.STATE_FISHING_CATCH,
        Fisherman.STATE_NO_FISHING_OK,
    ],
    get_transform(),
)

prepareTime = 5
input("Press enter to start botting in 5 seconds...")
for i in range(5):
    print(prepareTime - i, "...")
    sleep(1)

fishermanBot.start()
