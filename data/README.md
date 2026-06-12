# Data

This open-source release does not include training data, cached dataset files, or local experiment outputs.

## Public datasets

- MNIST / FashionMNIST / CIFAR-10 can be downloaded automatically by `torchvision`.

## Split index files

The original project may use pre-generated client split index files such as:

- `iid_cifar.npy`
- `non_iid_cifar.npy`
- `iid_fashion_mnist.npy`
- `non_iid_fashion_mnist.npy`

If these files are not included in your public release, you need to regenerate client partitions before running the full training pipeline.

## Not included

- downloaded raw datasets
- cached preprocessing files
- local benchmark outputs
- private or temporary experiment artifacts
