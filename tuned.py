import numpy as np
from afuncs import Sigmoid
import animate_loss as animations


def main():
    # XOR dataset
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])
    activation = Sigmoid()
    SHAPE = (2, 1_000, 1)
    
    # Seed for reproducibility
    np.random.seed(42)

    # Initialize weights and biases
    W1 = np.random.randn(SHAPE[0], SHAPE[1])
    W2 = np.random.randn(SHAPE[1], SHAPE[2])
    B1 = np.random.randn(SHAPE[1])
    B2 = np.random.randn(SHAPE[2])

    # parameters
    learning_rate = 0.1
    epochs = 100_000

    # Training
    O: np.ndarray
    losses = [] * epochs
    for epoch in range(epochs):
        # Forward Propagation
        H1 = activation(X @ W1 + B1)  
        O = activation(H1 @ W2 + B2)  

        # Calculate the error
        error = y - O
        loss = np.mean(np.abs(error))
        if epoch % 1000 == 0:
            print(f"Epoch {epoch} - Error: {loss}")

        # Backward Propagation
        d_O = error * activation.grad(H1 @ W2 + B2)
        d_H1 = d_O @ W2.T * activation.grad(X @ W1 + B1)

        # Update weights and biases
        W2 += H1.T @ d_O * learning_rate
        B2 += np.sum(d_O, axis=0) * learning_rate

        W1 += X.T @ d_H1 * learning_rate
        B1 += np.sum(d_H1, axis=0) * learning_rate

        losses.append(loss)

    # Testing the trained network
    print("\nTrained Perceptron Output (After training):")
    print(O.round())  # Round output to 0 or 1
    animations.plot(epochs, np.array(losses))


if __name__ == "__main__":
    main()
