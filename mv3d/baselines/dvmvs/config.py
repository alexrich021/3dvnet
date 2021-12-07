import time


class Config:
    use_wandb = False

    # training settings
    train_image_width = 256
    train_image_height = 256
    train_min_depth = 0.25
    train_max_depth = 20.0
    train_n_depth_levels = 64
    train_minimum_pose_distance = 0.125
    train_maximum_pose_distance = 0.325
    train_crawl_step = 3
    train_subsequence_length = None
    train_predict_two_way = None
    train_freeze_batch_normalization = False
    train_data_pipeline_workers = 8
    train_epochs = 100000
    train_print_frequency = 5000
    train_validate = True
    train_seed = int(round(time.time()))

    # test settings
    test_image_width = 320
    test_image_height = 256
    test_distortion_crop = 0
    test_perform_crop = False
    test_visualize = False
    test_n_measurement_frames = 2
    test_keyframe_buffer_size = 30
    test_keyframe_pose_distance = 0.1
    test_optimal_t_measure = 0.15
    test_optimal_R_measure = 0.0

    # SET THESE: selTRAINING FOLDER LOCATIONS
    dataset = '/home/alex/Desktop/scannet'
    train_run_directory = "/home/alex/Desktop/cloned_paper_code/deep-video-mvs/training-runs"

    # SET THESE: TESTING FOLDER LOCATIONS
    # for run-testing-online.py (evaluate a single scene, WITHOUT keyframe indices, online selection)
    # test_online_scene_path = "/home/alex/Desktop/cloned_paper_code/deep-video-mvs/sample-data/hololens-dataset/000"
    test_online_scene_path = "/home/alex/Desktop/scannet/scans_test/scene0707_00"
    # for run-testing.py (evaluate all available scenes, WITH pre-calculated keyframe indices)
    # test_offline_data_path = "/home/ardaduz/Workspace/git/deep-video-mvs/sample-data"
    test_offline_data_path = "/home/alex/Desktop/scannet/scans_test"
    # below give a dataset name like tumrgbd, i.e. folder or None
    # if None, all datasets will be evaluated given that
    # their keyframe index files are in Config.test_offline_data_path/indices folder
    test_dataset_name = "hololens-dataset"  # or None

    test_result_folder = "/home/alex/Desktop/out"
