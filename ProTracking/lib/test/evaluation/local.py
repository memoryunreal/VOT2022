from lib.test.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.

    settings.davis_dir = ''
    settings.got10k_lmdb_path = '/home/VOT2022/vot2022/votrgbd/ProTracking/data/got10k_lmdb'
    settings.got10k_path = '/home/VOT2022/vot2022/votrgbd/ProTracking/data/got10k'
    settings.got_packed_results_path = ''
    settings.got_reports_path = ''
    settings.lasot_lmdb_path = '/home/VOT2022/vot2022/votrgbd/ProTracking/data/lasot_lmdb'
    settings.lasot_path = '/home/VOT2022/vot2022/votrgbd/ProTracking/data/lasot'
    settings.network_path = '/home/VOT2022/vot2022/votrgbd/ProTracking/test/networks'    # Where tracking networks are stored.
    settings.nfs_path = ''
    settings.otb_path = ''
    settings.prj_dir = '/home/VOT2022/vot2022/votrgbd/ProTracking'
    settings.result_plot_path = '/home/VOT2022/vot2022/votrgbd/ProTracking/test/result_plots'
    settings.results_path = '/home/VOT2022/vot2022/votrgbd/ProTracking/test/tracking_results'    # Where to store tracking results
    settings.save_dir = '/home/VOT2022/vot2022/votrgbd/ProTracking'
    settings.segmentation_path = '/home/VOT2022/vot2022/votrgbd/ProTracking/test/segmentation_results'
    settings.tn_packed_results_path = ''
    settings.tpl_path = ''
    settings.trackingnet_path = '/home/VOT2022/vot2022/votrgbd/ProTracking/data/trackingNet'
    settings.uav_path = ''
    settings.vot20rgbd_path = '/home/VOT2022/vot2022/votrgbd/ProTracking/data/VOT20_RGBD'
    settings.vot_path = '/home/VOT2022/vot2022/votrgbd/ProTracking/data/VOT2019'
    settings.youtubevos_dir = ''

    return settings

