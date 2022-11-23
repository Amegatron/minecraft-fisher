from util import get_transform, get_classes
from dataset import FishermanDatasetPreparer
import sys

if len(sys.argv) < 2:
    print("Usage:\n")
    print("    python prepare.py <screenshot_dir> [<result_dir>]\n")
    print("If <result_dir> is not specified, it is considered to be <screenshot_dir>/_cache")
    exit(1)

img_dir = sys.argv[1]
result_dir = img_dir + '_cache'

if len(sys.argv) > 2:
    result_dir = sys.argv[2]

dataset_preparer = FishermanDatasetPreparer()

print("Preparing dataset...")
print("Original dir: %s" % img_dir)
print("Result dir: %s" % result_dir)
new_images = dataset_preparer.prepare(img_dir, result_dir, get_classes(), get_transform())
print("Done. %d new images processed." % new_images)
