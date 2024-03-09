% Fixed input data
X = rand(1, 100); % Random input data
Y = 2*X + 1 + 0.1*randn(1, 100); % Transformation: Y = 2*X + 1 + noise

% Initialize and train the model
input_size = size(X, 1);
hidden_size1 = 20;
hidden_size2 = 10;
hidden_size3 = 10;
output_size = size(Y, 1);

model = DeepNeuralNetwork(input_size, hidden_size1, hidden_size2, hidden_size3, output_size);
learning_rate = 0.01;
num_epochs = 1000;

model.train(X, Y, learning_rate, num_epochs);

% Make predictions
Y_pred = model.forward(X);

% Display results
disp('Predicted output vector:');
disp(Y_pred);
