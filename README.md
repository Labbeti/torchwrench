# torchwrench

<center>

<a href="https://www.python.org/">
    <img alt="Python" src="https://img.shields.io/badge/-Python 3.9+-blue?style=for-the-badge&logo=python&logoColor=white">
</a>
<a href="https://github.com/Labbeti/torchwrench/actions">
    <img alt="Build" src="https://img.shields.io/github/actions/workflow/status/Labbeti/torchwrench/test.yaml?branch=main&style=for-the-badge&logo=github">
</a>
<a href='https://torchwrench.readthedocs.io/'>
    <img src='https://readthedocs.org/projects/torchwrench/badge/?version=stable&style=for-the-badge' alt='Documentation Status' />
</a>
<a href="https://pytorch.org/get-started/locally/">
    <img alt="PyTorch" src="https://img.shields.io/badge/-PyTorch 1.10+-ee4c2c?style=for-the-badge&logo=pytorch&logoColor=white">
</a>

Collection of functions and modules to help development in PyTorch.

</center>


## Installation

With pip:
```bash
pip install torchwrench
```

With uv:
```bash
uv add torchwrench
```

The main requirement is **[PyTorch](https://pytorch.org/)**.

To check if the package is installed and show the package version, you can use the following command in your terminal:
```bash
torchwrench-info
```

This library works on all Python versions **>=3.9**, all PyTorch versions **>= 1.10**, and on **Linux, Mac and Windows** systems.

## Examples

`torchwrench` functions and modules can be used like `torch` ones. The default acronym for `torchwrench` is `tw`.

### Label conversions
Supports **multiclass** labels conversions between probabilities, classes indices, classes names and onehot encoding.

```python
import torchwrench as tw

probs = tw.as_tensor([[0.9, 0.1], [0.4, 0.6]])
names = tw.probs_to_name(probs, idx_to_name={0: "Cat", 1: "Dog"})
# ["Cat", "Dog"]
```

This package also supports **multilabel** labels conversions between probabilities, classes multi-indices, classes multi-names and multihot encoding.

```python
import torchwrench as tw

multihot = tw.as_tensor([[1, 0, 0], [0, 1, 1], [0, 0, 0]])
indices = tw.multihot_to_indices(multihot)
# [[0], [1, 2], []]
```

### Typing

```python
import torchwrench as tw

x1 = tw.as_tensor([1, 2])
print(isinstance(x1, tw.Tensor2D))  # False
x2 = tw.as_tensor([[1, 2], [3, 4]])
print(isinstance(x2, tw.Tensor2D))  # True
```

```python
import torchwrench as tw

x1 = tw.as_tensor([1, 2], dtype=tw.int)
print(isinstance(x1, tw.SignedIntegerTensor))  # True

x2 = tw.as_tensor([1, 2], dtype=tw.long)
print(isinstance(x2, tw.SignedIntegerTensor))  # True

x3 = tw.as_tensor([1, 2], dtype=tw.float)
print(isinstance(x3, tw.SignedIntegerTensor))  # False
```

### Padding

```python
import torchwrench as tw

x = tw.rand(10, 3, 1)
padded = tw.pad_dim(x, target_length=5, dim=1, pad_value=-1)
# x2 has shape (10, 5, 1), padded with -1
```

```python
import torchwrench as tw

tensors = [tw.rand(10, 2), tw.rand(5, 3), tw.rand(0, 5)]
padded = tw.pad_and_stack_rec(tensors, pad_value=0)
# padded has shape (3, 10, 5), padded with 0
```

### Masking

```python
import torchwrench as tw

x = tw.as_tensor([3, 1, 2])
mask = tw.lengths_to_non_pad_mask(x, max_len=4)
# Each row i contains x[i] True values for non-padding mask
# tensor([[True, True, True, False],
#         [True, False, False, False],
#         [True, True, False, False]])
```

```python
import torchwrench as tw

x = tw.as_tensor([1, 2, 3, 4])
mask = tw.as_tensor([True, True, False, False])
result = tw.masked_mean(x, mask)
# result contains the mean of the values marked as True: 1.5
```

### Others tensors manipulations!

```python
import torchwrench as tw

x = tw.as_tensor([1, 2, 3, 4])
result = tw.insert_at_indices(x, indices=[0, 2], values=5)
# result contains tensor with inserted values: tensor([5, 1, 2, 5, 3, 4])
```

```python
import torchwrench as tw

perm = tw.randperm(10)
inv_perm = tw.get_inverse_perm(perm)

x1 = tw.rand(10)
x2 = x1[perm]
x3 = x2[inv_perm]
# inv_perm are indices that allow us to get x3 from x2, i.e. x1 == x3 here
```

### Pre-compute datasets to HDF files

Here is an example of pre-computing spectrograms of torchaudio `SPEECHCOMMANDS` dataset, using `pack_dataset` function:

```python
from torchaudio.datasets import SPEECHCOMMANDS
from torchaudio.transforms import Spectrogram
from torchwrench import nn
from torchwrench.extras.hdf import pack_to_hdf

speech_commands_root = "path/to/speech_commands"
packed_root = "path/to/packed_dataset.hdf"

dataset = SPEECHCOMMANDS(speech_commands_root, download=True, subset="validation")
# dataset[0] is a tuple, contains waveform and other metadata

class MyTransform(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.spectrogram_extractor = Spectrogram()

    def forward(self, item):
        waveform = item[0]
        spectrogram = self.spectrogram_extractor(waveform)
        return (spectrogram,) + item[1:]

pack_to_hdf(dataset, packed_root, MyTransform())
```

Then you can load the pre-computed dataset using `HDFDataset`:
```python
from torchwrench.extras.hdf import HDFDataset

packed_root = "path/to/packed_dataset.hdf"
packed_dataset = HDFDataset(packed_root)
packed_dataset[0]  # == first transformed item, i.e. transform(dataset[0])
```

<!--
## Extras requirements
`torchwrench` also provides additional modules when some specific package are already installed in your environment.
All extras can be installed with `pip install torchwrench[extras]`

- If `tensorboard` is installed, the function `load_event_file` can be used. It is useful to load manually all data contained in an tensorboard event file.
- If `numpy` is installed, the classes `NumpyToTensor` and  `ToNumpy` can be used and their related function. It is meant to be used to compose dynamic transforms into `Sequential` module.
- If `h5py` is installed, the function `pack_to_hdf` and class `HDFDataset` can be used. Can be used to pack/read dataset to HDF files, and supports variable-length sequences of data.
- If `pyyaml` is installed, the functions `to_yaml` and `load_yaml` can be used. -->


## Contact
Maintainer:
- [Étienne Labbé](https://labbeti.github.io/) "Labbeti": labbeti.pub@gmail.com
