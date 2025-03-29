# train.py
import torch
from tqdm import tqdm
import wandb
from config import PROJECT_NAME, EPOCHS, LEARNING_RATE, SKIP_INDEX, RELATION_NUM, DEVICE, SAVED_MODEL_NAME, HIDDEN_DIM, OUTPUT_DIM
from utils import evaluate, mst_parsing, count_parameters
from models import model_initializing
import os


def start_wandb(experiment_name, wandb_config=None):
    if not wandb_config:
        wandb_config = {
            "learning_rate": LEARNING_RATE,
            "epochs": EPOCHS,
            "batch_size": 32,
            "model": "xlm-roberta-base",
            "output_dim": 256,
            "num_relations": RELATION_NUM
        }
    wandb.init(project=PROJECT_NAME, name=experiment_name, config=wandb_config)
    return wandb


def train(model, data, experiment_name, save_model=False):
    # Initialize Weights & Biases (wandb) for experiment tracking
    start_wandb(experiment_name)

    # Lists to store training loss and validation accuracy for each epoch
    val_accuracies = []
    train_losses = []

    # Fetch the dataloaders for training and validation sets
    train_dataloader = data["train"]
    val_loader = data["validation"]

    # Define the loss function for multi-class classification
    # Define loss functions for multi-class classification
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=SKIP_INDEX)  # For arc prediction
    loss_fn_rel = torch.nn.CrossEntropyLoss(ignore_index=SKIP_INDEX)  # For relation prediction

    # Define the optimizer with all relevant model parameters
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=LEARNING_RATE
    )

    # Main training loop
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss_tensor = torch.zeros(1, device=DEVICE)

        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{EPOCHS}"):
            # Move batch data to device
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            head_labels = batch['head'].to(DEVICE)        # Ground-truth head indices
            deprel_ids = batch['deprel_ids'].to(DEVICE)   # Ground-truth relation labels

            # Forward pass
            outputs = model(input_ids, attention_mask)
            arc_scores = outputs['arc_scores']  # Shape: (batch_size, seq_len, seq_len)
            rel_scores = outputs['rel_scores']  # Shape: (batch_size, seq_len, seq_len, num_relations)

            # Compute arc loss
            loss_arc = loss_fn(arc_scores.view(-1, arc_scores.shape[-1]), head_labels.view(-1))

            # Compute relation loss (using scores for true heads)
            batch_size, seq_len = head_labels.shape
            batch_idx = torch.arange(batch_size)[:, None].expand(-1, seq_len).to(DEVICE)
            dep_idx = torch.arange(seq_len)[None, :].expand(batch_size, -1).to(DEVICE)
            rel_scores_for_true_heads = rel_scores[batch_idx, dep_idx, head_labels, :]  # Shape: (batch_size, seq_len, num_relations)
            loss_rel = loss_fn_rel(rel_scores_for_true_heads.view(-1, RELATION_NUM), deprel_ids.view(-1))

            # Total loss
            loss = loss_arc + loss_rel
            epoch_loss_tensor += loss.detach()

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Compute average training loss
        epoch_loss = epoch_loss_tensor.item()
        avg_loss = epoch_loss / len(train_dataloader)
        train_losses.append(avg_loss)
        print(f"Epoch {epoch + 1}, Training Loss: {avg_loss:.4f}")

        # Validation
        end = epoch == (EPOCHS - 1)
        metrics = evaluate(model, val_loader)  # Updated evaluate function returns a dict
        val_uas = metrics['UAS']
        val_las = metrics['LAS']
        val_accuracies.append(val_uas)  # For backward compatibility, store UAS
        print(f"Epoch {epoch + 1}, Validation UAS: {val_uas:.4f}, LAS: {val_las:.4f}")

        uas_mst = mst_parsing(model, val_loader, DEVICE)  # Still computes UAS only
        print(f"Unlabeled Attachment Score (UAS) with MST: {uas_mst:.4f}")

        # Log metrics to wandb
        wandb.log({
            "epoch": epoch + 1,
            "training_loss": avg_loss,
            "validation_UAS": val_uas,
            "validation_LAS": val_las,
            "UAS_MST": uas_mst
        })

    if save_model:
        checkpoint_path = SAVED_MODEL_NAME
        torch.save(model.state_dict(), checkpoint_path)

    # Finalize wandb run
    wandb.finish()

    return model

def train_extended_models(extended_model_name, experiment_name, dataset):
    base_model = model_initializing("base", hidden_dim=HIDDEN_DIM, output_dim=OUTPUT_DIM, relation_num=RELATION_NUM)
    base_model.load_state_dict(torch.load(SAVED_MODEL_NAME))

    pfeiffer_extended_model = model_initializing(extended_model_name, trained_base_model=base_model)
    count_parameters(pfeiffer_extended_model)   
    model = train(pfeiffer_extended_model, dataset, experiment_name, save_model=False)
    return model


def test_model2(model, test_loader, checkpoint_path=None):
    """
    Evaluates the model on the test dataset, optionally loads a checkpoint, and logs the accuracy.

    Args:
        model (torch.nn.Module): The trained PyTorch model.
        test_loader (DataLoader): DataLoader for the test dataset.
        device (str): Device to perform computations on ('cuda' or 'cpu').
        checkpoint_path (str, optional): Path to the saved model checkpoint. If None, runs on the initialized model.

    Returns:
        float: Test accuracy of the model on the test dataset.
    """
    # Check if a checkpoint exists and load it
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading model checkpoint from {checkpoint_path}...")
        model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))

    # Move the model to the appropriate device
    model.to(DEVICE)
    model.eval()  # Set the model to evaluation mode

    # Evaluate the model on the test set
    test_accuracy = evaluate(model, test_loader)  # Assume the `evaluate` function is defined elsewhere
    test_uas = test_accuracy['UAS']
    test_las = test_accuracy['LAS']
    print(f"Test UAS: {test_uas:.4f}, LAS: {test_las:.4f}")

    mst_uas = mst_parsing(model, test_loader, DEVICE)
    print(f"Unlabeled Attachment Score (UAS): {mst_uas:.4f}")


    return test_accuracy
