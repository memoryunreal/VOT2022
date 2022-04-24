from collections import namedtuple
import importlib
from lib.test.evaluation.data import SequenceList

DatasetInfo = namedtuple('DatasetInfo', ['module', 'class_name', 'kwargs'])

pt = "lib.test.evaluation.%sdataset"  # Useful abbreviations to reduce the clutter

dataset_dict = dict(
    trackingnet=DatasetInfo(module=pt % "trackingnet", class_name="TrackingNetDataset", kwargs=dict()),
    got10k_train=DatasetInfo(module=pt % "got10k", class_name="GOT10KDataset", kwargs=dict(split='train')),
    got10k_test=DatasetInfo(module=pt % "got10k", class_name="GOT10KDataset", kwargs=dict(split='test')),
    got10k_val=DatasetInfo(module=pt % "got10k", class_name="GOT10KDataset", kwargs=dict(split='val')),
    got10k_ltrval=DatasetInfo(module=pt % "got10k", class_name="GOT10KDataset", kwargs=dict(split='ltrval')),
    lasot=DatasetInfo(module=pt % "lasot", class_name="LaSOTDataset", kwargs=dict(split='test')),
    lasot_train=DatasetInfo(module=pt % "lasot", class_name="LaSOTDataset", kwargs=dict(split='train')),
    lasot_lmdb=DatasetInfo(module=pt % "lasot_lmdb", class_name="LaSOTlmdbDataset", kwargs=dict(split="test"),),
    lasot_train_lmdb=DatasetInfo(module=pt % "lasot_lmdb", class_name="LaSOTlmdbDataset", kwargs=dict(split="train"), ),
    vot20rgbd=DatasetInfo(module=pt % "vot20rgbd", class_name="VOT20RGBDDataset", kwargs=dict()),
)


def load_dataset(name: str):
    """ Import and load a single dataset."""
    name = name.lower()
    dset_info = dataset_dict.get(name)
    if dset_info is None:
        raise ValueError('Unknown dataset \'%s\'' % name)

    m = importlib.import_module(dset_info.module)
    dataset = getattr(m, dset_info.class_name)(**dset_info.kwargs)  # Call the constructor
    return dataset.get_sequence_list()


def get_dataset(*args):
    """ Get a single or set of datasets."""
    dset = SequenceList()
    for name in args:
        dset.extend(load_dataset(name))
    return dset