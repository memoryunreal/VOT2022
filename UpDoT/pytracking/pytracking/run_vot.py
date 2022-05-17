import os
import sys
import argparse

env_path = os.path.join(os.path.dirname(__file__), '..')
if env_path not in sys.path:
    sys.path.append(env_path)

from pytracking.evaluation import Tracker


def run_vot2022rgbd(tracker_name, tracker_param, run_id=None, debug=0, visdom_info=None):
    tracker = Tracker(tracker_name, tracker_param, run_id)
    tracker.run_vot2022rgbd(debug, visdom_info)

def run_vot2022d(tracker_name, tracker_param, ax_tracker_param, run_id=None, debug=0, visdom_info=None):
    tracker = Tracker(tracker_name, tracker_param, ax_tracker=ax_tracker_param, run_id=None)
    tracker.run_vot2022d(debug, visdom_info)

def main():
    parser = argparse.ArgumentParser(description='Run VOT.')
    parser.add_argument('--tracker_name', type=str, default="dimp")
    parser.add_argument('--tracker_param', type=str, default="dimp50")
    parser.add_argument('--ax_tracker_param', type=str, default="dimp_ax")
    parser.add_argument('--run_id', type=int, default=None)
    parser.add_argument('--debug', type=int, default=0, help='Debug level.')
    parser.add_argument('--use_visdom', type=bool, default=True, help='Flag to enable visdom')
    parser.add_argument('--visdom_server', type=str, default='127.0.0.1', help='Server for visdom')
    parser.add_argument('--visdom_port', type=int, default=8097, help='Port for visdom')
    parser.add_argument('--dtype', type=str, default='d')
    parser.add_argument('--gpuid', type=str, default="0")

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid
    print(args.ax_tracker_param)
    visdom_info = {'use_visdom': args.use_visdom, 'server': args.visdom_server, 'port': args.visdom_port}
    if args.dtype == 'rgbd':
        run_vot2022rgbd(args.tracker_name, args.tracker_param, args.run_id, args.debug, visdom_info)
    else:
        run_vot2022d(args.tracker_name, args.tracker_param, ax_tracker_param=args.ax_tracker_param, run_id=args.run_id, debug=args.debug, visdom_info=visdom_info)


if __name__ == '__main__':
    main()
