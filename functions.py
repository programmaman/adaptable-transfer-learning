import torch
import torch.nn.functional as F
import random

def compute_gradient(params, data, loss_fn):
    """
    Computes the gradient of the loss function with respect to the parameters.
    """
    loss = loss_fn(params, data)
    grad = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
    return grad[0]  # Assuming single parameter tensor


def cosine_similarity(grad1, grad2):
    """
    Computes cosine similarity between two gradient tensors.
    """
    dot_product = torch.dot(grad1.view(-1), grad2.view(-1))
    norm1 = torch.norm(grad1)
    norm2 = torch.norm(grad2)
    return dot_product / (norm1 * norm2 + 1e-8)  # Add small value to prevent division by zero


def g(similarity, weights):
    """
    Weighting function using similarity and weights.
    """
    return F.softmax(similarity * weights, dim=0)  # Softmax to normalize weights


def loss_target(params, data, task_type="regression"):
    """
    Computes the supervised loss for the target task.
    """
    predictions = model_forward(params, data)
    labels = data["labels"].to(params.device)
    if task_type == "regression":
        return torch.mean((predictions - labels) ** 2)
    elif task_type == "classification":
        return F.cross_entropy(predictions, labels)


def loss_auxiliary(params, data):
    """
    Computes the loss for auxiliary tasks.
    """
    predictions = model_forward(params, data)
    pseudo_labels = data["pseudo_labels"].to(params.device)
    return torch.mean((predictions - pseudo_labels) ** 2)


def update_parameters(params, loss, learning_rate):
    """
    Updates model parameters using gradient descent.
    """
    grad = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
    with torch.no_grad():
        params -= learning_rate * grad[0]
    return params


def update_weights(weights, loss, learning_rate):
    """
    Updates weights for auxiliary tasks using gradient descent.
    """
    grad = torch.autograd.grad(loss, weights, retain_graph=True, allow_unused=True)
    with torch.no_grad():
        weights -= learning_rate * grad[0]
    return weights


def split_dataset(data):
    """
    Splits data into training and meta-validation sets.
    """
    split_point = int(len(data["features"]) * 0.8)
    train_data = {
        "features": data["features"][:split_point],
        "labels": data["labels"][:split_point]
    }
    meta_data = {
        "features": data["features"][split_point:],
        "labels": data["labels"][split_point:]
    }
    return train_data, meta_data


def model_forward(params, data):
    """
    Simulates a forward pass through the GNN.
    """
    adjacency_matrix = data["adjacency_matrix"].to(params.device)
    features = data["features"].to(params.device)
    return F.relu(torch.matmul(adjacency_matrix, torch.matmul(features, params)))


def initialize_model_parameters(dim):
    """
    Initializes model parameters (theta) for the GNN.
    """
    params = torch.empty(dim, device="cuda" if torch.cuda.is_available() else "cpu")
    return torch.nn.init.xavier_uniform_(params)


def initialize_weights(num_tasks):
    """
    Initializes weights for tasks (including target and auxiliary tasks).
    """
    return torch.ones(num_tasks, device="cuda" if torch.cuda.is_available() else "cpu")


def get_batch(Ds_sup, Ds_aux, batch_size=32):
    """
    Fetches a batch of data for supervised and auxiliary tasks.
    """
    sup_batch = random.sample(Ds_sup, batch_size)
    aux_batch = [random.sample(aux, batch_size) for aux in Ds_aux]
    return sup_batch, aux_batch
