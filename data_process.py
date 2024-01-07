import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import torchvision
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torchvision.transforms.functional as TF
import numpy as np

class WIDERFaceDataset(Dataset):
    """
    A custom dataset class for the WIDER Face dataset, used for face detection.

    Key Components:
    - __init__: Initializes dataset with directory, dataset split ('train'/'val'), and optional transformations.
    - parse_wider_txt: Reads and parses annotation text files to extract image paths and bounding box labels.
    - __len__ and __getitem__: Standard PyTorch Dataset methods to enable indexing and length retrieval.
    """
    def __init__(self, data_root, split, transform=None):
        assert split in ['train', 'val'], f"split must be in ['train', 'val'], got {split}"
        self.data_root = data_root
        self.split = split
        self.transform = transform
        self.img_paths, self.labels = self.parse_wider_txt()

    import os

    def parse_wider_txt(self):
        txt_path = os.path.join(self.data_root, 'wider_face_split', f'wider_face_{self.split}_bbx_gt.txt')
        img_root = os.path.join(self.data_root, f'WIDER_{self.split}', 'images')

        img_paths = []
        labels = []

        with open(txt_path, "r") as f:
            lines = f.readlines()
            file_name_line, num_boxes_line, box_annotation_line = True, False, False
            num_boxes, box_counter = 0, 0
            current_labels = []
            for line in lines:
                line = line.rstrip()
                if file_name_line:
                    img_path = os.path.join(img_root, line)
                    if os.path.exists(img_path):  # Check if the image file exists
                        img_paths.append(img_path)
                        file_name_line = False
                        num_boxes_line = True
                    else:  # Skip to the next file name line if the image file does not exist
                        file_name_line = True
                        num_boxes_line = False
                        box_annotation_line = False
                        current_labels = []
                        continue
                elif num_boxes_line:
                    num_boxes = int(line)
                    num_boxes_line = False
                    box_annotation_line = True if num_boxes > 0 else True
                elif box_annotation_line:
                    box_counter += 1
                    line_values = line.split(" ")
                    current_labels.append([int(x) for x in line_values[:4]])
                    if box_counter >= num_boxes:
                        if img_path in img_paths:  # Add labels only if the image path was added
                            labels.append(current_labels)
                        current_labels = []
                        box_counter = 0
                        file_name_line = True
        return img_paths, labels

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        # Original WIDERFaceDataset Logic
        img_path = self.img_paths[idx]
        image = Image.open(img_path).convert("RGB")
        original_size = image.size
        boxes = self.labels[idx] # Original format: [x, y, w, h]

        numpy_image = np.array(image)

        # Convert boxes from [x, y, w, h] to [x_min, y_min, x_max, y_max]
        boxes = [[x, y, x + w, y + h] for x, y, w, h in boxes]
        boxes = torch.tensor(boxes, dtype=torch.float32)

        num_objs = len(boxes)
        # print(num_objs)
        masks = torch.zeros((num_objs, image.height, image.width), dtype=torch.uint8)
        for i, box in enumerate(boxes):
            # Convert bounding boxes to masks
            # box format is [x_min, y_min, x_max, y_max]
            x_min, y_min, x_max, y_max = box
            masks[i, int(y_min):int(y_max), int(x_min):int(x_max)] = 1

        labels = torch.ones((num_objs,), dtype=torch.int64)

        image_id = torch.tensor([idx])
        if num_objs > 0:
            # Calculate area for each box
            area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        else:
            area = torch.zeros((0,), dtype=torch.float32)
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transform is not None:
            image, target["boxes"], target["masks"] = transform_image_and_target(image, target["boxes"],
                                                                                 target["masks"], self.transform)

        return image, target


def visualize_sample(image, boxes):
    """Visualize an image with bounding boxes.

    Args:
        image: The image to be visualized.
        boxes: A list of bounding boxes, each box is [x_min, y_min, x_max, y_max].
    """
    # Display the image
    plt.imshow(image)
    ax = plt.gca()

    # Add each bounding box to the plot
    for box in boxes:
        x_min, y_min, x_max, y_max = box
        width, height = x_max - x_min, y_max - y_min
        rect = patches.Rectangle((x_min, y_min), width, height, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

    plt.show()


def my_collate_fn(batch):
    # Separate images and targets
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]

    # Stack images into a tensor
    images = torch.stack(images)

    return images, targets

# def transform_image_and_target(image, target, transform):
#     transformed_image = transform(image)
#     transformed_target = {}
#
#     if isinstance(transformed_image, torch.Tensor):
#         # Tensor shape: [C, H, W]
#         new_height, new_width = transformed_image.shape[1], transformed_image.shape[2]
#     else:
#         raise TypeError("Transformed image is not a torch.Tensor. It's a {}".format(type(transformed_image)))
#
#     original_width, original_height = image.size
#     scale_x, scale_y = new_width / original_width, new_height / original_height
#
#     # Transform bounding boxes, if present in target
#     if 'boxes' in target and target['boxes'].numel() > 0:
#         boxes = target['boxes']
#         new_boxes = []
#         for box in boxes:
#             x, y, x_max, y_max = box
#             new_x, new_y = x * scale_x, y * scale_y
#             new_x_max, new_y_max = x_max * scale_x, y_max * scale_y
#             new_boxes.append([new_x, new_y, new_x_max, new_y_max])
#         transformed_target['boxes'] = torch.tensor(new_boxes, dtype=torch.float32)
#
#     # Adjust masks, if present
#     if 'masks' in target and target['masks'].numel() > 0:
#         masks = target['masks']
#         new_masks = []
#         for mask in masks:
#             # Convert tensor mask to PIL Image for transformation
#             mask = transforms.functional.to_pil_image(mask)
#             # Apply transform to the PIL Image mask
#             transformed_mask = transform(mask)
#             # transformed_mask is already a tensor, so no need to convert it again
#             new_masks.append(transformed_mask)
#         transformed_target['masks'] = torch.stack(new_masks)
#
#     for key in target:
#         if key not in ['boxes', 'masks']:
#             transformed_target[key] = target[key]
#
#     return transformed_image, transformed_target


def transform_image_and_target(image, boxes, masks, transform):
    # Store the original size of the image
    original_size = image.size

    # Apply the transform to the image.
    # Assuming the transform includes ToTensor and Normalize
    image = transform(image)

    # Get the new size of the image
    new_size = [image.shape[2], image.shape[1]]  # Tensor shape: [C, H, W]

    # Scale the bounding boxes accordingly.
    scale_x, scale_y = new_size[0] / original_size[0], new_size[1] / original_size[1]
    new_boxes = boxes * torch.tensor([scale_x, scale_y, scale_x, scale_y])

    # Transform the masks
    # Resize mask to new image size and ensure it's still a binary mask
    new_masks = torch.zeros((len(masks), new_size[1], new_size[0]), dtype=torch.uint8)
    for i, mask in enumerate(masks):
        new_mask = transforms.functional.resize(mask.unsqueeze(0).float(),
                                                (new_size[1], new_size[0]),
                                                interpolation=transforms.InterpolationMode.NEAREST)
        new_masks[i] = new_mask.squeeze().byte()

    return image, new_boxes, new_masks


if __name__ == "__main__":
    # Dataset and DataLoader instantiation for training split
    transform = transforms.Compose(
        [transforms.Resize((224, 224)),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    dataset = WIDERFaceDataset(data_root='./data', split='train', transform=transform)
    data_loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=1, collate_fn=my_collate_fn)

    # Get a batch of data
    batch = next(iter(data_loader))
    images, targets = batch

    # Debug: Print the structure and content of the batch
    print("Debug: Batch Structure")
    print("Number of elements in the batch:", len(batch))
    print("Images shape:", images.shape)
    print("Targets structure:", len(targets))

    # Dive deeper into the first target's details
    if len(targets) > 0:
        first_target = targets[0]
        print("\nDebug: First Target in Batch")
        for key, value in first_target.items():
            print(f"{key}: {value.shape if hasattr(value, 'shape') else type(value)}")

    # Visualize the first image in the batch along with its bounding boxes
    image_np = images[0].numpy().transpose((1, 2, 0))
    image_np = (image_np * 255).astype('uint8')
    boxes = first_target["boxes"].tolist() if "boxes" in first_target else []

    print("Number of boxes:", len(boxes))
    visualize_sample(image_np, boxes)
