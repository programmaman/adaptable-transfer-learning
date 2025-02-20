import torch
from functions import (
    compute_gradient,
    cosine_similarity,
    g,
    loss_target,
    loss_auxiliary,
    update_parameters,
    update_weights,
    split_dataset,
    initialize_weights,
    initialize_model_parameters,
    get_batch,
)

# Inputs
Ds_sup = []  # Supervised training data for the target task
Ds_aux = [[] for _ in range(5)]  # Auxiliary tasks training data (5 tasks for example)
alpha = 0.01  # Learning rate for GNN
beta = 0.001  # Learning rate for weighting model
N = 100  # Max iterations
K = 5  # Number of auxiliary tasks

# Define dimensions for the GNN parameters and initialize them
feature_dim = 10  # Example feature dimension
theta = initialize_model_parameters(feature_dim)  # Shared GNN parameters
w = initialize_weights(K + 1)  # Weights for target and auxiliary tasks
sim = torch.ones(K + 1, device=theta.device)  # Task similarity scores (initialized to 1)

# Training loop
for t in range(N):
    # Fetch a batch of training data
    batch_sup, batch_aux = get_batch(Ds_sup, Ds_aux)

    # Step 1: Optimize the weighting model via meta-learning
    train_sup, meta_sup = split_dataset(batch_sup)  # Split target task data into train/meta

    # Compute gradient similarities for all auxiliary tasks
    grad_sup = compute_gradient(theta, train_sup, lambda p, d: loss_target(p, d, "regression"))
    for i in range(K):
        grad_aux = compute_gradient(theta, batch_aux[i], loss_auxiliary)
        sim[i + 1] = cosine_similarity(grad_sup, grad_aux)  # Cosine similarity

    # Calculate joint loss using current weights
    joint_loss = g(sim[0], w[0]) * loss_target(theta, train_sup, "regression")
    for i in range(K):
        joint_loss += g(sim[i + 1], w[i + 1]) * loss_auxiliary(theta, batch_aux[i])

    # Update GNN parameters temporarily with joint loss
    theta_temp = update_parameters(theta, joint_loss, alpha)

    # Update weighting model using meta loss on meta_sup
    meta_loss = loss_target(theta_temp, meta_sup, "regression")
    w = update_weights(w, meta_loss, beta)

    # Step 2: Optimize the GNN with updated weights
    # Recompute gradient similarities with updated weights
    grad_sup = compute_gradient(theta, train_sup, lambda p, d: loss_target(p, d, "regression"))
    for i in range(K):
        grad_aux = compute_gradient(theta, batch_aux[i], loss_auxiliary)
        sim[i + 1] = cosine_similarity(grad_sup, grad_aux)

    # Calculate joint loss with updated weights
    joint_loss = g(sim[0], w[0]) * loss_target(theta, train_sup, "regression")
    for i in range(K):
        joint_loss += g(sim[i + 1], w[i + 1]) * loss_auxiliary(theta, batch_aux[i])

    # Update GNN parameters using joint loss
    theta = update_parameters(theta, joint_loss, alpha)

    # Print progress
    if t % 10 == 0:
        print(f"Iteration {t}: Joint Loss = {joint_loss.item():.4f}")

# Final output
print("Training complete.")
