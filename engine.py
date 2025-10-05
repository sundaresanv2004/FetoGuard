import torch
from tqdm.notebook import tqdm
from utils import dice_score  # We will add this function to utils.py


def train_step(model, dataloader, loss_fn, optimizer, device):
    """Performs a single training step for one epoch."""
    model.train()
    train_loss = 0.0

    progress_bar = tqdm(dataloader, desc="Training")

    for images, masks in progress_bar:
        # Move data to the target device
        images = images.to(device)
        masks = masks.to(device, dtype=torch.long)

        # 1. Forward pass
        outputs = model(images)

        # 2. Calculate loss
        loss = loss_fn(outputs, masks)
        train_loss += loss.item()

        # 3. Optimizer zero grad, backward pass, and optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update progress bar
        progress_bar.set_postfix(loss=loss.item())

    # Calculate average loss over all batches
    avg_train_loss = train_loss / len(dataloader)
    return avg_train_loss


def test_step(model, dataloader, loss_fn, device, num_classes):
    """Performs a single evaluation step on the test set."""
    model.eval()
    test_loss = 0.0
    test_dice = 0.0

    progress_bar = tqdm(dataloader, desc="Validating")

    with torch.no_grad():
        for images, masks in progress_bar:
            # Move data to the target device
            images = images.to(device)
            masks = masks.to(device, dtype=torch.long)

            # 1. Forward pass
            outputs = model(images)

            # 2. Calculate loss
            loss = loss_fn(outputs, masks)
            test_loss += loss.item()

            # 3. Calculate Dice score
            # First, convert model outputs (logits) to predicted class indices
            preds = torch.argmax(outputs, dim=1)
            dice = dice_score(preds, masks, num_classes=num_classes)
            test_dice += dice

            # Update progress bar
            progress_bar.set_postfix(loss=loss.item(), dice=dice)

    # Calculate average loss and Dice score over all batches
    avg_test_loss = test_loss / len(dataloader)
    avg_test_dice = test_dice / len(dataloader)

    return avg_test_loss, avg_test_dice