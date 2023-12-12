from quilt_texture import QuiltTexture
from texture_transfer import TextureTransfer
from skimage import io
import matplotlib.pyplot as plt
import numpy as np
import argparse
import sys
import logging

parser = argparse.ArgumentParser()
parser.add_argument("-M", "--model", required=True, choices=["quilt_texture", "texture_transfer"], default=None,
                    help="Select only quilt_texture or texture_transfer")
parser.add_argument("-i", "--image_path", required=True, type=str, help="path of image you want to quilt")
parser.add_argument("-p", "--patch_length", type=int, default=40, help="block size in pixels")
parser.add_argument("-qs", "--quilt_size_shape", type=int, nargs=2)
parser.add_argument("-np", "--num_patches", type=int, nargs=2, default=(10, 10),
                    help="number of patches in (rows, cols)")
parser.add_argument("-m", "--mode", default="cut", choices=["random", "neighboring", "cut"])
parser.add_argument("-s", "--sequence", action="store_true", help="display each patch in every step")
parser.add_argument("-f", "--output_file", type=str, default=None, help="output file name")
parser.add_argument("-plt", "--plot", action="store_true", help="display result")
parser.add_argument("-ti", "--target_image")
parser.add_argument("-tri", "--transfer_iter", default=2)

args = parser.parse_args()

if __name__ == "__main__":
    image_path = args.image_path
    patch_length = args.patch_length
    shape = args.quilt_size_shape
    num_patches = args.num_patches
    mode = args.mode
    sequence = args.sequence
    image = io.imread(image_path)

    try:
        if args.model == "quilt_texture":
            quilt_texture = QuiltTexture(texture=image, patch_length=patch_length)

            if args.quilt_size_shape:
                result = quilt_texture.quilt_size(shape=shape, mode=mode, sequence=sequence)
            elif args.num_patches:
                result = quilt_texture.quilt(num_patches=num_patches, mode=mode, sequence=sequence)

        elif args.model == "texture_transfer":
            if not args.target_image:
                raise ValueError("Target image is missing for texture transfer.")
            target_image = io.imread(args.target_image)
            texture_transfer = TextureTransfer(texture=image, target=target_image, patch_length=patch_length)
            result = texture_transfer.transfer_iter(args.transfer_iter, mode=args.mode)

    except ValueError as ve:
        print(f"ValueError: {ve}")
        print("Please check the input parameters.")
        sys.exit(1)
    except Exception as e:
        logging.exception("An unexpected error occurred:")
        print("An unexpected error occurred. Please check the logs for more details.")
        sys.exit(1)

    result = (result * 255).astype(np.uint8)

    if args.output_file:
        io.imsave(args.output_file, result)

    if args.plot:
        plt.imshow(result)
        plt.axis('off')
        plt.show()
