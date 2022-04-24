from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path as osp
import sys


def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)


this_dir = osp.dirname(__file__)

prj_dir = osp.join(this_dir, '../..')
add_path(prj_dir)
lib_dir = osp.join(prj_dir, 'lib')
add_path(lib_dir)