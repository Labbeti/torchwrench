from pythonwrench.jsonl import dump_jsonl as dump_jsonl
from pythonwrench.jsonl import load_jsonl as load_jsonl
from torch.serialization import *

from .common import as_builtin as as_builtin
from .csv import dump_csv as dump_csv
from .csv import load_csv as load_csv
from .dump_fn import dump as dump
from .dump_fn import save as save
from .json import dump_json as dump_json
from .json import load_json as load_json
from .load_fn import load as load
from .pickle import dump_pickle as dump_pickle
from .pickle import load_pickle as load_pickle
from .torch import dump_torch as dump_torch
from .torch import load_torch as load_torch
