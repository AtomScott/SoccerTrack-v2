import os
import shutil
import argparse
from sklearn.model_selection import train_test_split
import yaml
from loguru import logger
def generate_yaml(config_path, output_dir, train_images_path, val_images_path, test_images_path):
    """
    Generates a new YAML file for the dataset split.

    Args:
        config_path (str): Path to the original YOLO config file.
        output_dir (str): Directory to save the new YAML file.
        train_images_path (str): Path to the training images directory.
        val_images_path (str): Path to the validation images directory.
        test_images_path (str): Path to the testing images directory.
    """
    # Read the original YAML file
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    # Create a new config with only 'names' and 'nc'
    new_config = {key: config[key] for key in ['names', 'nc'] if key in config}

    # Add paths for train, val, and test images
    new_config['train'] = train_images_path
    new_config['val'] = val_images_path
    new_config['test'] = test_images_path

    # Write the new config to a YAML file
    new_yaml_path = os.path.join(output_dir, 'data.yaml')
    with open(new_yaml_path, 'w') as file:
        yaml.safe_dump(new_config, file, sort_keys=False)

    logger.info(f"Generated new YAML file at {new_yaml_path}")

def split_dataset(image_path, label_path, output_dir, train, test, val, config_path, random_state):
    """
    Splits the YOLO dataset into train, test, and validation sets and organizes them into specified structure.

    Args:
        image_path (str): Directory containing the images.
        label_path (str): Directory containing the labels.
        output_dir (str): Directory to save the split dataset.
        train (float|int): Ratio or number of the dataset to be used for training.
        test (float|int): Ratio or number of the dataset to be used for testing.
        val (float|int): Ratio or number of the dataset to be used for validation.
        config_path (str): Path to the YOLO config file.
        random_state (int): Seed used by the random number generator.
    """
    # Create directories for the splits if they don't exist
    for split in ['train', 'valid', 'test']:
        os.makedirs(os.path.join(output_dir, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, split, 'labels'), exist_ok=True)

    # List all image files
    images = [file for file in os.listdir(image_path) if file.endswith('.jpg') or file.endswith('.png')]

    # Split dataset
    train = int(train) if train >= 1 else train
    test = int(test) if test >= 1 else test
    val = int(val) if val >= 1 else val
    train_val_images, test_images = train_test_split(images, test_size=test, random_state=random_state)
    train_images, val_images = train_test_split(train_val_images, train_size=train, test_size=val, random_state=random_state)

    # Function to copy files to their respective directories
    def copy_files(files, split):
        for file in files:
            shutil.copy(os.path.join(image_path, file), os.path.join(output_dir, split, 'images', file))
            annotation_file = file.rsplit('.', 1)[0] + '.txt'
            shutil.copy(os.path.join(label_path, annotation_file), os.path.join(output_dir, split, 'labels', annotation_file))

    # Copy files to their respective directories
    for images, dataset_name in [(train_images, 'train'), (test_images, 'test'), (val_images, 'valid')]:
        copy_files(images, dataset_name)
        logger.info(f'Copied {len(images)} images to {dataset_name} directory')

    # Paths for the split directories
    train_images_path = os.path.join(output_dir, 'train', 'images')
    val_images_path = os.path.join(output_dir, 'valid', 'images')
    test_images_path = os.path.join(output_dir, 'test', 'images')

    # Call generate_yaml at the end of the split_dataset function
    generate_yaml(config_path, output_dir, train_images_path, val_images_path, test_images_path)


def main():
    parser = argparse.ArgumentParser(description="Split a YOLO dataset into train, test, and validation sets.")
    parser.add_argument("--image_path", type=str, required=True, help="Directory containing the images.")
    parser.add_argument("--label_path", type=str, required=True, help="Directory containing the labels.")
    parser.add_argument("--config_path", type=str, required=True, help="Path to the YOLO config file.")
    parser.add_argument("--train_ratio", type=float, default=0.7, help="Ratio of the dataset to be used for training.")
    parser.add_argument("--test_ratio", type=float, default=0.2, help="Ratio of the dataset to be used for testing.")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Ratio of the dataset to be used for validation.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the split dataset.")
    parser.add_argument("--random_state", type=int, default=42, help="Random state for the train_test_split function.")
    args = parser.parse_args()

    split_dataset(args.image_path, args.label_path, args.output_dir, args.train_ratio, args.test_ratio, args.val_ratio, args.config_path, args.random_state)

if __name__ == "__main__":
    main()

# Example usage:
# python split.py --image_path /path/to/images --label_path /path/to/labels --config_path /path