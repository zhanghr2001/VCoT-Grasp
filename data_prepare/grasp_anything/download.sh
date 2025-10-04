# export HF_ENDPOINT=https://hf-mirror.com
cd /opt/data/private/MLLM_dataset/grasp_anything

huggingface-cli download airvlab/Grasp-Anything --repo-type dataset --local-dir ./ --include "image_part_*"
huggingface-cli download airvlab/Grasp-Anything --repo-type dataset --local-dir ./ --include "grasp_label_positive.zip"
huggingface-cli download airvlab/Grasp-Anything --repo-type dataset --local-dir ./ --include "grasp_label_negative.zip"
huggingface-cli download airvlab/Grasp-Anything --repo-type dataset --local-dir ./ --include "scene_description.zip"
huggingface-cli download airvlab/Grasp-Anything --repo-type dataset --local-dir ./ --include "mask.zip"

cat image_part_aa image_part_ab > image.zip
unzip -q image.zip -d ./
unzip -q grasp_label_positive.zip -d ./
unzip -q grasp_label_negative.zip -d ./
unzip -q scene_description.zip -d ./
unzip -q mask.zip -d ./
