import argparse
from PIL import Image

def crop_image(img_path, left_crop, top_crop, right_crop, bottom_crop):
    # Open the image file
    img = Image.open(img_path)

    # Get the original image size
    original_width, original_height = img.size

    # Define the new edges by cropping the specified amount from each side
    left = left_crop
    top = top_crop
    right = original_width - right_crop
    bottom = original_height - bottom_crop

    # Crop the image
    cropped_img = img.crop((left, top, right, bottom))

    # Save the cropped image
    cropped_img_path = img_path.replace('.jpg', '_cropped.jpg')
    cropped_img.save(cropped_img_path)

    return cropped_img_path

def main():
    parser = argparse.ArgumentParser(description='Crop an image by specified amounts from each side.')
    parser.add_argument('img_path', type=str, help='Path to the image file')
    parser.add_argument('--left', type=int, default=0, help='Amount to crop from the left side of the image')
    parser.add_argument('--top', type=int, default=0, help='Amount to crop from the top of the image')
    parser.add_argument('--right', type=int, default=0, help='Amount to crop from the right side of the image')
    parser.add_argument('--bottom', type=int, default=0, help='Amount to crop from the bottom of the image')
    
    args = parser.parse_args()

    cropped_img_path = crop_image(args.img_path, args.left, args.top, args.right, args.bottom)
    print(f'Cropped image saved to {cropped_img_path}')

if __name__ == '__main__':
    main()
