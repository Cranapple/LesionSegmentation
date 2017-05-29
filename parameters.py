import numpy

patch_size = 25
output_size = 25 - 16 #Change depending on number of conv.
batch_size = 1000
depth1 = 30
depth2 = 40
depth3 = 50
num_hidden = 150
kernel_size = 5
train_size = 100000
valid_size = 10000
database_size = train_size + valid_size

def shuffle_in_unison(a, b):
    assert len(a) == len(b)
    shuffled_a = numpy.empty(a.shape, dtype=a.dtype)
    shuffled_b = numpy.empty(b.shape, dtype=b.dtype)
    permutation = numpy.random.permutation(len(a))
    for old_index, new_index in enumerate(permutation):
        shuffled_a[new_index] = a[old_index]
        shuffled_b[new_index] = b[old_index]
    return shuffled_a, shuffled_b