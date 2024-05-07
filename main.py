from model import NeuralNetwork
import numpy as np

# Load training and testing data
def load_data(file_path):
    with open(file_path, 'r') as file:
        data = file.readlines()
    return [list(line.strip()) for line in data]

def encode_labels(labels):
    label_map = {'#': [1, -1, -1, -1], '.': [-1, 1, -1, -1], 'o': [-1, -1, 1, -1], '@': [-1, -1, -1, 1]}
    return [label_map[label] for label in labels]

def main():
    # Load training data
    train_data = load_data('HW3_Training-1.txt')
    train_labels = encode_labels([pixel for line in train_data for pixel in line])
    train_inputs = np.array(train_labels)

    # Load testing data
    test_data = load_data('HW3_Testing-1.txt')
    test_labels = encode_labels([pixel for line in test_data for pixel in line])
    test_inputs = np.array(test_labels)

    # Model parameters
    input_size = len(train_inputs[0])
    hidden_size = 16
    output_size = len(train_labels[0])
    learning_rate = 0.000001
    sigmoid_param = 1
    epochs = 100
    initial_weights = None  # Weights will be initialized randomly

    # Initialize neural network
    nn = NeuralNetwork(input_size, hidden_size, output_size, learning_rate, sigmoid_param, initial_weights)

    # Train the model
    nn.train(train_inputs, train_labels, epochs)

    # Test the model
    predictions = nn.predict(test_inputs)
    predicted_labels = [np.argmax(prediction) for prediction in predictions]
    true_labels = [np.argmax(label) for label in test_labels]

    # Calculate accuracy
    accuracy = np.mean(np.array(predicted_labels) == np.array(true_labels))
    print("Accuracy:", accuracy)

if __name__ == "__main__":
    main()
