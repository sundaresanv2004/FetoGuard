import torch
import config
from data_setup import create_dataloaders
from utils import get_device


def main():
    """Main function to run the data loading and preparation process."""
    # Set seed for reproducibility
    torch.manual_seed(config.RANDOM_SEED)

    # 1. Setup device
    if config.DEVICE == "auto":
        device = get_device()
    else:
        device = torch.device(config.DEVICE)
        print(f"Using device: {device}")

    # 2. Create DataLoaders
    train_loader, test_loader = create_dataloaders(
        anno_path=config.ANNO_PATH,
        img_path=config.IMG_PATH,
        batch_size=config.BATCH_SIZE,
        train_split_ratio=config.TRAIN_SPLIT_RATIO,
        image_size=config.IMAGE_SIZE,
        random_seed=config.RANDOM_SEED,
        num_workers=config.NUM_WORKERS
    )

    print("\nDataLoaders created successfully.")
    print(f"Number of batches in train_loader: {len(train_loader)}")
    print(f"Number of batches in test_loader: {len(test_loader)}")

    # 3. Example: Check a batch from the train_loader
    try:
        images, masks = next(iter(train_loader))
        print("\nSuccessfully fetched one batch from the train_loader:")
        print(f"  Images batch shape: {images.shape}")  # Should be [BATCH_SIZE, 3, IMAGE_SIZE, IMAGE_SIZE]
        print(f"  Masks batch shape:  {masks.shape}")  # Should be [BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE]

        # In a real training loop, you would move data to the device
        # images = images.to(device)
        # masks = masks.to(device)
        # ... training steps ...

    except StopIteration:
        print("Train loader is empty, cannot fetch a batch.")
    except Exception as e:
        print(f"An error occurred while fetching a batch: {e}")


if __name__ == "__main__":
    main()