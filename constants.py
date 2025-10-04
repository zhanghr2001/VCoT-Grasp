paligemma_model_id = "google/paligemma2-3b-mix-224"
pretrained_paligemma_dir = "pretrained"

action_seq_len = 5
action_with_binned_angle_seq_len = 22
angle_bins = 18

grasp_anything_planar_grasp_train_csv_path = "split/grasp_anything_match/train.csv"
grasp_anything_planar_grasp_test_seen_csv_path = "split/grasp_anything_match/test_seen.csv"
grasp_anything_planar_grasp_test_unseen_csv_path = "split/grasp_anything_match/test_unseen.csv"

grasp_anything_rgb_root = "/baishuanghao/mllm_data/grasp_anything/image"
grasp_anything_mask_root = "/baishuanghao/mllm_data/grasp_anything/mask"
grasp_anything_description_root = "/baishuanghao/mllm_data/grasp_anything/scene_description"
grasp_anything_planar_grasp_root = "/baishuanghao/mllm_data/grasp_anything/grasp_label_positive"

real_image_root = "/baishuanghao/mllm_data/grasp_real-world-v1/planar_data_process"
real_grasp_data_path = "/baishuanghao/mllm_data/grasp_real-world-v1/grasp_real-world-v1_annotations_with_grasp_labels.xml"
real_bbox_data_path = "/baishuanghao/mllm_data/grasp_real-world-v1/grasp_real-world-v1_annotations_with_object_detection.xml"
