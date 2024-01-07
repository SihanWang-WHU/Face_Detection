from data_process import WIDERFaceDataset
from data_process import my_collate_fn
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from model import get_model_instance_segmentation
from reference.engine import train_one_epoch

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def main():
    num_classes = 2
    transform = transforms.Compose(
        [transforms.Resize((256, 256)),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    dataset = WIDERFaceDataset(data_root='./data', split='train', transform=transform)
    dataset_test = WIDERFaceDataset(data_root='./data', split='val', transform=transform)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        num_workers=0,
        collate_fn=my_collate_fn
    )
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=2,
        shuffle=False,
        num_workers=0,
        collate_fn=my_collate_fn
    )

    model = get_model_instance_segmentation(num_classes)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params,
        lr=0.001,
        momentum=0.9,
        weight_decay=0.0005
    )
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=100,
        gamma=0.5
    )

    num_epochs = 2

    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        # update the learning rate
        lr_scheduler.step()
    torch.save(model.state_dict(), 'model.pth')


if __name__ == '__main__':
    main()