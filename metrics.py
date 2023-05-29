from skimage import metrics
from pathlib import Path
from PIL import Image
import numpy as np

a = [f for f in Path("C:\Study\Diplom\DiplomApp\HR_original").iterdir() if f.is_file()]

b = Image.open("C:\Study\Diplom\DiplomApp\HR_original\\482403_out.jpg")

import tkinter as tk
from tkinter import filedialog

root = tk.Tk()
root.withdraw()

file_path = filedialog.askopenfilename()

name = Path(file_path).stem
ext = Path(file_path).suffix
print(ext)

def calculate_metrics(name):
    org = Image.open(file_path)
    shape = org.size
    org = np.asarray(org)

    sr = np.asarray(Image.open(f"C:\Study\Diplom\DiplomApp\HR_original\{name}_out{ext}").resize(shape))
    bcb = np.asarray(Image.open(f"C:\Study\Diplom\DiplomApp\HR_original\{name}_bcb{ext}").resize(shape))

    print("_________________________________")
    print("FILENAME", name)
    print("PNSE:",metrics.peak_signal_noise_ratio(np.asarray(org), np.asarray(sr)))
    print("SSIM:",metrics.structural_similarity(np.asarray(org), np.asarray(sr), channel_axis=2, data_range=255))

    print("\n Bicubic:__________")
    print("PNSE:",metrics.peak_signal_noise_ratio(np.asarray(org), np.asarray(bcb)))
    print("SSIM:",metrics.structural_similarity(np.asarray(org), np.asarray(bcb), channel_axis=2, data_range=255))
    print("______________")



calculate_metrics(name)


print(a[i])
original = Image.open(a[i])
shape = original.size
sr = Image.open(a[i+1]).resize(shape)
print(np.asarray(original).shape, np.asarray(sr).shape)
print("Orignal file:", a[i], "SR: ", a[i+1], "BCB: ", a[i+2])
print("PNSE:",metrics.peak_signal_noise_ratio(np.asarray(original), np.asarray(sr)))
print("SSIM:",metrics.structural_similarity(np.asarray(original), np.asarray(sr), channel_axis=2, data_range=255))
