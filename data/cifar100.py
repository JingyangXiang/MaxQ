import torch
import torchvision
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms


class CIFAR100:
    def __init__(self, args):
        super(CIFAR100, self).__init__()

        data_root = args.data

        use_cuda = torch.cuda.is_available()

        # Data loading code
        kwargs = {"num_workers": args.workers, "pin_memory": True} if use_cuda else {}

        normalize = transforms.Normalize(
            mean=[0.491, 0.482, 0.447], std=[0.247, 0.243, 0.262]
        )

        train_dataset = torchvision.datasets.CIFAR100(
            root=data_root,
            train=True,
            download=True,
            transform=transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ]
            ),
        )
        self.train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs
        )

        test_dataset = torchvision.datasets.CIFAR100(
            root=data_root,
            train=False,
            download=True,
            transform=transforms.Compose([transforms.ToTensor(), normalize]),
        )
        self.val_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=args.batch_size, shuffle=False, **kwargs
        )
