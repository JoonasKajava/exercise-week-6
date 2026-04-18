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
if rank == 0:
    print("size:", size, flush=True)
print("rank:", rank, flush=True)

start = time.time()

chunks = []
results = []
if rank == 0:
    target = abspath(sys.argv[1])

    files = [Path(join(target, f)) for f in os.listdir(target)]
    print("total files:", len(files), flush=True)
    chunks = np.array_split(files, size)
    print("chunks:", len(chunks), "size:", size, flush=True)

def analyse(chunk):
    for path in chunk:
        image = np.load(path)
        image = (image - np.min(image)) / (np.max(image) - np.min(image))
        blobs = blob_log(image, max_sigma=30, num_sigma=10, threshold=0.1)
        results.append(len(blobs))
        print(f"{path.name}: ", len(blobs), flush=True)

data = comm.scatter(chunks, root=0)

print(f"rank: {rank} scatter:", len(data), flush=True)

analyse(data)

gatheredResults = comm.gather(results, root=0)

if rank == 0:
    flat = np.array([x for xs in gatheredResults for x in xs])
    print("Min blobs:", flat.min(), flush=True)
    print("Mean blobs:", flat.mean(), flush=True)
    print("Max blobs:", flat.max(), flush=True)
    end = time.time()
    length = end - start
    print("Analysis took: ", length, "seconds", flush=True)