import os
import sys
import time
from os.path import abspath, join
from pathlib import Path

import multiprocessing as mp
import numpy as np
from skimage.feature import blob_log

from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
print("rank:", rank)

start = time.time()

target = abspath(sys.argv[1])

files = [Path(join(target, f)) for f in os.listdir(target)]

values: list[int] = []
names = []

count = 0

chunks = np.array_split(files, mp.cpu_count())

def analyse(chunk):
    for path in chunk:
        image = np.load(path)
        normalizedData = (image - np.min(image)) / (np.max(image) - np.min(image))
        # e = entropy(normalizedData, disk(5))
        # e = exposure.equalize_hist(normalizedData)
        e = blob_log(normalizedData, max_sigma=30, num_sigma=10, threshold=0.1)
        print(f"{path.name}: ", len(e))

processes = []
for chunk in chunks:
    new_process = mp.Process(target=analyse, args=[chunk])
    processes.append(new_process)

for p in processes:
    p.start()

for p in processes:
    p.join()
    # for path in chunk:
        # print(f"{count}/{len(files)}", sum(e[0]))
        # count += 1
# for other in files:
#     other_image = np.load(other)
#     s = ssim(image,other_image, data_range=other_image.max() - other_image.min())
#
#     print(f"{count}/{len(files)*len(files)}:", s)
#     count += 1

# names.append(path.name)
# var = image.var()
# values.append(var)
# print(f"{path.name}: {var:.2f}")

# var_sum = sum(values)
#
# plt.figure(figsize=(10, 6))
# plt.bar(names, values)
# plt.xticks(rotation=45, ha="right")
# plt.ylim(min(values) - 1.0, max(values) + 1.0)
# plt.title("Variance of Sampled Images")
# plt.xlabel("Image Name")
# plt.ylabel("Variance")
# plt.figtext(0.05, 0.95, f"sum of variances: {var_sum:.2f}", fontsize=10)
# plt.savefig("analysis.png", bbox_inches="tight")
end = time.time()
length = end - start

print("Analysis took: ", length, "seconds")