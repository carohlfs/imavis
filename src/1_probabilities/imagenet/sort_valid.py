import os
import tarfile
import shutil
import scipy.io

# Unlike the ImageNet training dataset, the validation tar does not come
# pre-organized into folders based upon the human labels. We need to use
# two additional files to organize the images to be used by PyTorch.
# The source files are extracted from ILSVRC2012_devkit.tar.gz by
# probabilities.sh into the ../data/imagenet folder.
ground_truth_file = '../data/imagenet/ILSVRC2012_devkit_t12/data/ILSVRC2012_validation_ground_truth.txt'
meta_file = '../data/imagenet/ILSVRC2012_devkit_t12/data/meta.mat'
tar_file = '../tmp/imagenet/ILSVRC2012_img_val.tar'
output_dir = '../data/imagenet/val/'

# Load the ground truth labels
with open(ground_truth_file, 'r') as f:
    ground_truth = [int(line.strip()) for line in f]

# Load synset mappings from meta.mat
meta = scipy.io.loadmat(meta_file)

# Extract only the WordNet IDs (the second element in each tuple)
synsets = [str(synset[1][0]) for synset in meta['synsets'][:,0]]  # WordNet IDs

# Open the tarfile containing validation images
with tarfile.open(tar_file) as tar:
    members = tar.getmembers()  # Get all members (files) in the tar archive
    image_files = [m for m in members if m.isfile()]  # Filter to get only the image files (JPEGs)

    # Sort image files by their filenames (name attribute of TarInfo)
    image_files.sort(key=lambda x: x.name)

    if len(image_files) != len(ground_truth):
        raise ValueError("The number of image files does not match the number of ground truth labels!")

    # Process each image file
    for i, img_file in enumerate(image_files):
        label = ground_truth[i]  # Get the corresponding label from ground truth
        class_folder = os.path.join(output_dir, synsets[label - 1])  # Subtract 1 for 0-based index

        # Create class folder if it doesn't exist
        if not os.path.exists(class_folder):
            os.makedirs(class_folder)

        # Extract and move the image to the class folder
        extracted_path = tar.extractfile(img_file)  # Extract the image file
        dest = os.path.join(class_folder, os.path.basename(img_file.name))  # Destination in the class folder
        
        # Write the extracted image file to its destination
        with open(dest, 'wb') as out_file:
            shutil.copyfileobj(extracted_path, out_file)  # Write the image to the destination folder

        print(f"Moved {img_file.name} to {class_folder}")
print("Extraction and organization complete.")
