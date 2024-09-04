import os
import tarfile
import shutil

# Unlike the ImageNet training dataset, the validation tar does not come
# pre-organized into folders based upon the human labels. We need to use
# two additional files to organize the images to be used by PyTorch.
# The source files are extracted from ILSVRC2017_devkit.tar.gz by
# probabilities.sh into the ../tmp/imagenet folder.
ground_truth_file = '../tmp/imagenet/ILSVRC2015_clsloc_validation_ground_truth.txt'
mapping_file = '../tmp/imagenet/map_clsloc.txt'
tar_file = '../tmp/imagenet/ILSVRC2012_img_val.tar'
output_dir = '../data/imagenet/val/'

# Load the ground truth labels
with open(ground_truth_file, 'r') as f:
    ground_truth_labels = f.read().splitlines()

# Load the mapping of class IDs to labels
class_mapping = {}
with open(mapping_file, 'r') as f:
    for line in f:
        parts = line.split()
        class_mapping[parts[1]] = parts[0]  # mapping from label index to class label

# Open the tar file
with tarfile.open(tar_file, 'r') as tar:
    # Extract all files from the tar
    for i, member in enumerate(tar.getmembers()):
        # Get the class label from the ground truth
        class_id = ground_truth_labels[i]
        class_label = class_mapping[class_id]
        # Create the output directory if it doesn't exist
        class_dir = os.path.join(output_dir, class_label)
        os.makedirs(class_dir, exist_ok=True)
        # Extract the image to the correct directory
        tar.extract(member, path=class_dir)
        # Move the file to the correct directory (it may extract to an intermediate directory)
        extracted_file_path = os.path.join(class_dir, member.name)
        final_file_path = os.path.join(class_dir, os.path.basename(member.name))
        shutil.move(extracted_file_path, final_file_path)
        # Remove the intermediate directory if it exists
        intermediate_dir = os.path.join(class_dir, member.name.split('/')[0])
        if os.path.isdir(intermediate_dir):
            shutil.rmtree(intermediate_dir)

print("Extraction and organization complete.")
