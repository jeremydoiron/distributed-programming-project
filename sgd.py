from mpi4py import MPI
import numpy as np
import time

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Parameters
learning_rate = 0.01
num_epochs = 100
n_features = 1000

# Dataset
np.random.seed(42 + rank)
data_size_per_rank = 1_000_000
X = np.random.randn(data_size_per_rank, n_features)  # Features
y = np.random.randn(data_size_per_rank, 1)  # Labels

def compute_gradient(X, y, weights):
    predictions = X @ weights
    errors = predictions - y
    gradient = (X.T @ errors) / len(X)
    return gradient

def ring_allreduce(gradient):
    local_gradient = gradient.copy()
    start_time = time.time()
    for i in range(size - 1):
        send_rank = (rank + 1) % size
        recv_rank = (rank - 1 + size) % size

        send_buffer = local_gradient.copy()
        recv_buffer = np.empty_like(gradient)
        comm.Sendrecv(send_buffer, dest=send_rank, recvbuf=recv_buffer, source=recv_rank)

        local_gradient += recv_buffer
    comm.Barrier()
    elapsed_time = time.time() - start_time

    local_gradient /= size
    return local_gradient, elapsed_time

def tree_allreduce(gradient):
    local_gradient = gradient.copy()
    start_time = time.time()
    step = 1
    while step < size:
        partner = rank ^ step
        if partner < size:
            send_buffer = local_gradient.copy()
            recv_buffer = np.empty_like(local_gradient)
            comm.Sendrecv(send_buffer, dest=partner, recvbuf=recv_buffer, source=partner)

            local_gradient += recv_buffer
        step *= 2
    comm.Barrier()
    elapsed_time = time.time() - start_time

    local_gradient /= size
    return local_gradient, elapsed_time

def mpi_allreduce(gradient):
    local_gradient = np.empty_like(gradient)
    start_time = time.time()
    comm.Allreduce(gradient, local_gradient, op=MPI.SUM)
    comm.Barrier()
    elapsed_time = time.time() - start_time

    local_gradient /= size
    return local_gradient, elapsed_time

algos = [ring_allreduce, tree_allreduce, mpi_allreduce]

for algo in algos:
    weights = np.zeros((n_features, 1))
    total_time = 0

    for epoch in range(num_epochs):
        local_gradient = compute_gradient(X, y, weights)

        averaged_gradient, elapsed_time = algo(local_gradient)
        total_time += elapsed_time

        weights -= learning_rate * averaged_gradient

        if rank == 0 and epoch % 10 == 0:
            loss = np.mean((X @ weights - y)**2)
            print(f"Epoch {epoch}, Loss: {loss}, Algo: {algo.__name__}, Time: {elapsed_time:.4f}s")

    if rank == 0:
        final_weights = np.empty((size, n_features, 1))
    else:
        final_weights = None
    comm.Gather(weights, final_weights, root=0)

    if rank == 0:
        avg_time_per_epoch = total_time / num_epochs
        print(f"Final model parameters collected from all ranks with {algo.__name__}:", final_weights)
        print(f"Average time per epoch for {algo.__name__}: {avg_time_per_epoch:.4f}s")
