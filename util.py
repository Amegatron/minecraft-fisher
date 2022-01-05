import win32gui
import win32ui
import time
from ctypes import windll
from PIL import Image
from torchvision.transforms import Compose, CenterCrop, Resize, Grayscale


def get_state_filename():
    return "fisherman_state.pt"


def get_transform():
    return Compose(
        [
            CenterCrop(300),
            Resize(128),
            Grayscale(),
        ]
    )


def get_classes():
    return ["FishingIdle", "FishingCatch", "NoFishingOk"]


def find_window(title):
    searchResults = []

    def callback(hwnd, needle):
        windowTitle = win32gui.GetWindowText(hwnd)

        if needle in windowTitle:
            searchResults.append((hwnd, windowTitle))

    win32gui.EnumWindows(callback, title)

    return searchResults


'''
Taken from https://stackoverflow.com/a/24352388/2054918
'''
def grab_window_image(hwnd):
    left, top, right, bot = win32gui.GetClientRect(hwnd)
    w = right - left
    h = bot - top
    hwndDC = win32gui.GetWindowDC(hwnd)
    mfcDC = win32ui.CreateDCFromHandle(hwndDC)
    saveDC = mfcDC.CreateCompatibleDC()

    saveBitMap = win32ui.CreateBitmap()
    saveBitMap.CreateCompatibleBitmap(mfcDC, w, h)

    saveDC.SelectObject(saveBitMap)
    windll.user32.PrintWindow(hwnd, saveDC.GetSafeHdc(), 1)

    bmpinfo = saveBitMap.GetInfo()
    bmpstr = saveBitMap.GetBitmapBits(True)

    im = Image.frombuffer(
        'RGB',
        (bmpinfo['bmWidth'], bmpinfo['bmHeight']),
        bmpstr, 'raw', 'BGRX', 0, 1)

    win32gui.DeleteObject(saveBitMap.GetHandle())
    saveDC.DeleteDC()
    mfcDC.DeleteDC()
    win32gui.ReleaseDC(hwnd, hwndDC)

    return im


def train_model(model, epochs, loss_func, optimizer, train_data, test_data, device, target_acc=0.9, verbose=False):
    for epoch in range(epochs):
        train_loss, train_iters = 0, 0
        train_acc, train_pass = 0, 0
        start_time = time.time()

        model.train()
        for y, X in train_data:
            X = X.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            y_pred = model(X)
            l = loss_func(y_pred, y)
            l.backward()
            optimizer.step()
            train_loss += l.item()
            train_iters += 1
            train_acc += (y_pred.argmax(1) == y.argmax(1)).sum().item()
            train_pass += len(X)

        test_loss, test_iters = 0, 0
        test_acc, test_pass = 0, 0
        model.eval()
        for y, X in test_data:
            X = X.to(device)
            y = y.to(device)
            y_pred = model(X)
            l = loss_func(y_pred, y)
            test_loss += l.item()
            test_iters += 1
            test_acc += (y_pred.argmax(1) == y.argmax(1)).sum().item()
            test_pass += len(X)

        test_acc = test_acc / test_pass
        if verbose:
            vars = (epoch, time.time() - start_time, train_loss / train_iters,
                    test_loss / test_iters, train_acc / train_pass, test_acc)
            print("Epoch %d finished in %d s. Train loss: %f. Test loss: %f. Train acc: %f. Test acc: %f" % vars)

        if test_acc >= target_acc:
            print("Reached target accuracy of %f: %f" % (target_acc, test_acc))
            return
