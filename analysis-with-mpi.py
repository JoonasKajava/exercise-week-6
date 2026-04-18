import os
import sys
import time
from os.path import abspath, join
from pathlib import Path

import numpy as np
from skimage.feature import blob_log

from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
print("rank:", rank)

start = time.time()

chunks = []
results = []
if rank == 0:
    target = abspath(sys.argv[1])

    files = [Path(join(target, f)) for f in os.listdir(target)]
    print("total files:", len(files))
    chunks = np.array_split(files, size)
    print("chunks:", len(chunks), "size:", size)

def analyse(chunk):
    for path in chunk:
        image = np.load(path)
        normalizedData = (image - np.min(image)) / (np.max(image) - np.min(image))
        # e = entropy(normalizedData, disk(5))
        # e = exposure.equalize_hist(normalizedData)
        e = blob_log(normalizedData, max_sigma=30, num_sigma=10, threshold=0.1)
        results.append(e)
        print(f"{path.name}: ", len(e))

data = comm.scatter(chunks, root=0)

print(f"rank: {rank} scatter:", len(data))

analyse(data)

newData = comm.gather(results, root=0)
if rank == 0:
    print(f"rank: {rank} gather:", len(newData))

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

if rank == 0:
    print("Analysis took: ", length, "seconds")