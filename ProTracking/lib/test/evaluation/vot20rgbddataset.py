import numpy as np
from lib.test.evaluation.data import Sequence, BaseDataset, SequenceList
from lib.test.utils.load_text import load_text
import os


class VOT20RGBDDataset(BaseDataset):
    def __init__(self):
        super().__init__()
        self.base_path = os.path.join(self.env_settings.vot20rgbd_path, "sequences")
        self.sequence_list = self._get_sequence_list()

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_list])

    def _construct_sequence(self, sequence_name):
        anno_path = '{}/{}/groundtruth.txt'.format(self.base_path, sequence_name)
        ground_truth_rect = load_text(str(anno_path), delimiter=',', dtype=np.float64)

        frames_path = '{}/{}/color'.format(self.base_path, sequence_name)
        frame_list = [frame for frame in os.listdir(frames_path) if frame.endswith(".jpg")]
        frame_list = sorted(frame_list)
        frames_list = [os.path.join(frames_path, frame) for frame in frame_list]

        return Sequence(sequence_name, frames_list, 'vot20rgbd', ground_truth_rect.reshape(-1, 4))

    def __len__(self):
        return len(self.sequence_list)

    def _get_sequence_list(self):
        with open(os.path.join(self.base_path, "list.txt")) as f:
            sequence_list = f.read().splitlines()
        return sequence_list
