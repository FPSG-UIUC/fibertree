from fibertree import Codec
from boltons.cacheutils import LRU
import sys
from fibertree import Tensor
import time

# take an HFA tensor, convert it to compressed representation in python
def encodeSwoopTensorInFormat(tensor, descriptor, tensor_shape=None, cache_size=32):
    codec = Codec(tuple(descriptor), [True]*len(descriptor))

    # get output dict based on rank names
    rank_names = tensor.getRankIds()
    # print("encode tensor: rank names {}, descriptor {}".format(rank_names, descriptor))
    # TODO: move output dict generation into codec
    output = codec.get_output_dict(rank_names)
    # print("output dict {}".format(output))
    output_tensor = []
    for i in range(0, len(descriptor)+1):
            output_tensor.append(list())

    # print("encode, output {}".format(output_tensor))
    codec.encode(-1, tensor.getRoot(), tensor.getRankIds(), output, output_tensor, shape=tensor_shape)

    # name the fibers in order from left to right per-rank
    rank_idx = 0
    rank_names = ["root"] + tensor.getRankIds()
    # tensor_cache = dict()

    tensor_cache = LRU(max_size = cache_size)
    for rank in output_tensor:
        fiber_idx = 0
        for fiber in rank:
            fiber_name = "_".join([tensor.getName(), rank_names[rank_idx], str(fiber_idx)])
            fiber.setName(fiber_name)
            # fiber.printFiber()
            fiber.cache = tensor_cache
            fiber_idx += 1
        rank_idx += 1
    return output_tensor

# tensor is a 2d linearized tensor (one list per rank)
# dump all stats into output dict
def dumpAllStatsFromTensor(tensor, output, cache_output, name):
    for rank in tensor:
        for fiber in rank:
            fiber.dumpStats(output)
    cache_output[name + '_buffer_access'] = tensor[0][0].cache.hit_count
    cache_output[name + '_DRAM_access'] = tensor[0][0].cache.miss_count

# HFA reading in utils
def get_A_HFA(a_file):
    # read in inputs
    # jhu_len = 5157
    shape = 500 # TODO: take this as input
    # generate input frontier and tile it
    A_data = [0] * shape

    # read in frontier
    count = 1
    A_HFA = None
    if not a_file.endswith('.yaml'):
        with open(a_file, 'r') as f:
            for line in f:
                elt = int(line)
                A_data[elt] = count
                count += 1
        A_untiled = Tensor.fromUncompressed(["S"], A_data, name ="A")
        A_HFA = A_untiled.splitUniform(32, relativeCoords=True) # split S
        print("A untiled shape {}, tiled shape {}".format(A_untiled.getShape(), A_HFA.getShape()
        ))

    else: # already in pretiled yaml
        A_HFA = Tensor.fromYAMLfile(a_file)
        A_HFA.setName("A")
    return A_HFA

def get_B_HFA(b_file):
    print("reading tiled mtx from yaml")
    t0 = time.clock()
    B_HFA = Tensor.fromYAMLfile(b_file)
    t1 = time.clock() - t0
    print("read B from yaml in {} s".format(t1))
    # B_HFA.print()
    return B_HFA

def get_Z_HFA(B_shape):
    Z_data = [[0], [0]]
    Z_HFA = Tensor.fromUncompressed(["D1", "D0"], Z_data, shape=[B_shape[0], B_shape[2]],
            name="Z")
    return Z_HFA
