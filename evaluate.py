import torch
from data_process import WIDERFaceDataset
from data_process import my_collate_fn
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from model import get_model_instance_segmentation

def calculate_iou(pred_box, true_box):
    """
    Calculate the Intersection over Union (IoU) between two bounding boxes.
    Bounding box format: [x_min, y_min, x_max, y_max]
    """
    # Calculate intersection coordinates
    x_min_inter = max(pred_box[0], true_box[0])
    y_min_inter = max(pred_box[1], true_box[1])
    x_max_inter = min(pred_box[2], true_box[2])
    y_max_inter = min(pred_box[3], true_box[3])

    # Compute area of intersection
    inter_area = max(x_max_inter - x_min_inter, 0) * max(y_max_inter - y_min_inter, 0)

    # Compute areas of bounding boxes
    pred_area = (pred_box[2] - pred_box[0]) * (pred_box[3] - pred_box[1])
    true_area = (true_box[2] - true_box[0]) * (true_box[3] - true_box[1])

    # Compute union area
    union_area = pred_area + true_area - inter_area

    # Calculate IoU
    iou = inter_area / union_area if union_area != 0 else 0

    return iou

def update_confusion_matrix(conf_matrix, targets, outputs, iou_threshold=0.5):
    """
    Update the confusion matrix.
    conf_matrix: Confusion matrix, size num_classes x num_classes.
    targets: Ground truth labels.
    outputs: Model predictions.
    iou_threshold: IoU threshold.
    """
    for target, output in zip(targets, outputs):
        target_boxes = target['boxes']
        target_labels = target['labels']
        output_boxes = output['boxes']
        output_labels = output['labels']

        # For each predicted bounding box
        for i, pred_box in enumerate(output_boxes):
            pred_label = output_labels[i]
            max_iou = 0
            matched_label = None

            # Find the matching ground truth box
            for j, true_box in enumerate(target_boxes):
                true_label = target_labels[j]
                iou = calculate_iou(pred_box, true_box)
                if iou > max_iou:
                    max_iou = iou
                    matched_label = true_label

            # Update the confusion matrix based on IoU and class information
            if max_iou >= iou_threshold:
                # True Positive
                conf_matrix[matched_label, pred_label] += 1
            else:
                # False Positive
                conf_matrix[matched_label, pred_label] += 1

    return conf_matrix


def evaluate(model, val_data_loader, device, conf_matrix, iou_threshold=0.1):
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        for images, targets in val_data_loader:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            outputs = model(images)

            # Process each image in the batch
            for i, output in enumerate(outputs):
                output_boxes = output['boxes']
                output_labels = output['labels']

                target_boxes = targets[i]['boxes']
                target_labels = targets[i]['labels']

                # For each predicted bounding box, update the confusion matrix
                for pred_box, pred_label in zip(output_boxes, output_labels):
                    max_iou = 0
                    matched_label = None

                    # Find the matching ground truth box
                    for true_box, true_label in zip(target_boxes, target_labels):
                        iou = calculate_iou(pred_box.cpu().numpy(), true_box.cpu().numpy())
                        if iou > max_iou:
                            max_iou = iou
                            matched_label = true_label.item()

                    # Update the confusion matrix
                    if max_iou >= iou_threshold and matched_label is not None:
                        # True Positive
                        conf_matrix[matched_label, pred_label.item()] += 1
                    else:
                        # False Positive
                        conf_matrix[0, pred_label.item()] += 1  # Assuming 0 is the index for background/no-object class

    # After going through the data, return the updated confusion matrix
    return conf_matrix


if __name__ == '__main__':
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    num_classes = 2
    model = get_model_instance_segmentation(num_classes)
    model.to(device)
    model.load_state_dict(torch.load('model.pth'))
    model.eval()

    transform = transforms.Compose(
        [transforms.Resize((256, 256)),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    dataset_test = WIDERFaceDataset(data_root='./data', split='val', transform=transform)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=2,
        shuffle=False,
        num_workers=0,
        collate_fn=my_collate_fn
    )
    conf_matrix = torch.zeros(num_classes, num_classes, dtype=torch.int64)
    conf_matrix= evaluate(model, data_loader_test, device, conf_matrix)
    print(conf_matrix)