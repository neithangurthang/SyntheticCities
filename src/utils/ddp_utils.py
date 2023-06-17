# UTILITY FUNCTIONS FOR DATA DISTRIBUTED PARALLELIZATION
import torchvision.datasets as dset
from torch.utils.data import DataLoader, Dataset

def prepare_dataloader(dataset: Dataset, batch_size: int):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,
        sampler=DistributedSampler(dataset)
    )

def ddp_setup(rank, world_size):
    '''
    Args:
            rank: unique identifier for each process -> rank is given automatically by mp.spawn()
            world_size: total number of processes
    '''
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    init_process_group(backend='nccl', rank = rank, world_size = world_size)
    torch.cuda.set_device(rank)