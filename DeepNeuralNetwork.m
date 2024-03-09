classdef DeepNeuralNetwork < handle
    properties
        input_size
        hidden_size1
        hidden_size2
        hidden_size3
        output_size
        W1
        b1
        W2
        b2
        W3
        b3
        W4
        b4
    end
    
    methods
        function obj = DeepNeuralNetwork(input_size, hidden_size1, hidden_size2, hidden_size3, output_size)
            obj.input_size = input_size;
            obj.hidden_size1 = hidden_size1;
            obj.hidden_size2 = hidden_size2;
            obj.hidden_size3 = hidden_size3;
            obj.output_size = output_size;
            
            % Initialize weights and biases
            obj.W1 = randn(hidden_size1, input_size);
            obj.b1 = zeros(hidden_size1, 1);
            obj.W2 = randn(hidden_size2, hidden_size1);
            obj.b2 = zeros(hidden_size2, 1);
            obj.W3 = randn(hidden_size3, hidden_size2);
            obj.b3 = zeros(hidden_size3, 1);
            obj.W4 = randn(output_size, hidden_size3);
            obj.b4 = zeros(output_size, 1);
        end
        
        function Y_pred = forward(obj, X)
            Z1 = obj.W1 * X + obj.b1;
            A1 = relu(Z1);
            Z2 = obj.W2 * A1 + obj.b2;
            A2 = relu(Z2);
            Z3 = obj.W3 * A2 + obj.b3;
            A3 = relu(Z3);
            Z4 = obj.W4 * A3 + obj.b4;
            Y_pred = Z4;
        end
        
        function train(obj, X, Y, learning_rate, num_epochs)
            for epoch = 1:num_epochs
                % Forward propagation
                Z1 = obj.W1 * X + obj.b1;
                A1 = relu(Z1);
                Z2 = obj.W2 * A1 + obj.b2;
                A2 = relu(Z2);
                Z3 = obj.W3 * A2 + obj.b3;
                A3 = relu(Z3);
                Z4 = obj.W4 * A3 + obj.b4;
                Y_pred = Z4;                
                % Compute loss (mean squared error)
                loss = 0.5 * sum((Y_pred - Y).^2) / size(Y, 2);
                
                % Backpropagation
                dZ4 = Y_pred - Y;
                dW4 = (1 / size(Y, 2)) * dZ4 * A3';
                db4 = (1 / size(Y, 2)) * sum(dZ4, 2);
                dA3 = obj.W4' * dZ4;
                dZ3 = dA3 .* reluGradient(Z3);
                dW3 = (1 / size(Y, 2)) * dZ3 * A2';
                db3 = (1 / size(Y, 2)) * sum(dZ3, 2);
                dA2 = obj.W3' * dZ3;
                dZ2 = dA2 .* reluGradient(Z2);
                dW2 = (1 / size(Y, 2)) * dZ2 * A1';
                db2 = (1 / size(Y, 2)) * sum(dZ2, 2);
                dA1 = obj.W2' * dZ2;
                dZ1 = dA1 .* reluGradient(Z1);
                dW1 = (1 / size(Y, 2)) * dZ1 * X';
                db1 = (1 / size(Y, 2)) * sum(dZ1, 2);
                
                % Update parameters
                obj.W1 = obj.W1 - learning_rate * dW1;
                obj.b1 = obj.b1 - learning_rate * db1;
                obj.W2 = obj.W2 - learning_rate * dW2;
                obj.b2 = obj.b2 - learning_rate * db2;
                obj.W3 = obj.W3 - learning_rate * dW3;
                obj.b3 = obj.b3 - learning_rate * db3;
                obj.W4 = obj.W4 - learning_rate * dW4;
                obj.b4 = obj.b4 - learning_rate * db4;
                
                % Print progress
                if mod(epoch, 100) == 0
                    disp(['Epoch ', num2str(epoch), ', Loss: ', num2str(loss)]);
                end
            end
        end
    end
end

function y = relu(x)
    y = max(0, x);
end

function dx = reluGradient(x)
    dx = x > 0;
end
