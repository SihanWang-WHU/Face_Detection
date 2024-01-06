import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import matplotlib.patches as patches

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
                    img_paths.append(img_path)
                    file_name_line = False
                    num_boxes_line = True
                elif num_boxes_line:
                    num_boxes = int(line)
                    num_boxes_line = False
                    box_annotation_line = True if num_boxes > 0 else True
                elif box_annotation_line:
                    box_counter += 1
                    line_values = line.split(" ")
                    current_labels.append([int(x) for x in line_values[:4]])
                    if box_counter >= num_boxes:
                        labels.append(current_labels)
                        current_labels = []
                        box_counter = 0
                        file_name_line = True
        return img_paths, labels

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        image = Image.open(img_path).convert("RGB")
        boxes = self.labels[idx]

        if self.transform:
            image, boxes = transform_image_and_boxes(image, boxes, self.transform)

        return image, boxes

def visualize_sample(image, boxes):
    """ Visualize an image with bounding boxes """
    plt.imshow(image)
    ax = plt.gca()
    for box in boxes:
        x, y, width, height = box
        rect = patches.Rectangle((x, y), width, height, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
    plt.show()


def my_collate_fn(batch):
    images = []
    boxes = []

    for item in batch:
        images.append(item[0])
        boxes.append(torch.tensor(item[1]))

    images = torch.stack(images, dim=0)
    return images, boxes

def transform_image_and_boxes(image, boxes, transform):
    # Store the original size of the image
    original_size = image.size

    # Apply the transform to the image.
    image = transform(image)

    # Get the new size of the image
    new_size = [image.shape[2], image.shape[1]]  # Tensor shape: [C, H, W]

    # Scale the bounding boxes accordingly.
    new_boxes = []
    scale_x, scale_y = new_size[0] / original_size[0], new_size[1] / original_size[1]
    for box in boxes:
        x, y, width, height = box
        new_x, new_y, new_width, new_height = x * scale_x, y * scale_y, width * scale_x, height * scale_y
        new_boxes.append([new_x, new_y, new_width, new_height])

    return image, new_boxes


if __name__ == "__main__":
    # Dataset and DataLoader instantiation for training split
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    dataset = WIDERFaceDataset(data_root='./data', split='train', transform=transform)
    data_loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4, collate_fn=my_collate_fn)
    print(len(dataset.img_paths), len(dataset.labels))

    # Get a batch of data
    images, boxes = next(iter(data_loader))

    # Convert the first image in the batch to a numpy array and scale it back to [0,255]
    image_np = images[0].numpy().transpose((1, 2, 0))
    image_np = (image_np * 255).astype('uint8')

    # Visualize the first image in the batch along with its bounding boxes
    visualize_sample(image_np, boxes[0])