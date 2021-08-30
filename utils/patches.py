import torch
import torch.nn.functional as F

from typing import Union, Tuple

def images_to_patches(images: torch.Tensor, patch_size: Union[int, Tuple[int]]):
    patch_size = (patch_size, patch_size) if isinstance(patch_size, int) else patch_size
    patches = F.unfold(images, kernel_size=patch_size, padding=0, stride=patch_size)
    return patches

def patches_to_images(patches : torch.Tensor, output_size: Union[int, Tuple[int]], patch_size: Union[int, Tuple[int]]):
    patch_size = (patch_size, patch_size) if isinstance(patch_size, int) else patch_size
    images = F.fold(patches, output_size=output_size, kernel_size=patch_size, padding=0, stride=patch_size)
    return images