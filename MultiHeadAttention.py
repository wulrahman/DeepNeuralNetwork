import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the Transformer Encoder architecture
class MultiHeadAttention(nn.Module):
    def __init__(self, input_dim, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert input_dim % num_heads == 0, "Input dimension must be divisible by the number of heads"
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads

        self.W_q = nn.Linear(input_dim, input_dim)
        self.W_k = nn.Linear(input_dim, input_dim)
        self.W_v = nn.Linear(input_dim, input_dim)

        self.fc = nn.Linear(input_dim, input_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)

    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]

        Q = self.W_q(query)
        K = self.W_k(key)
        V = self.W_v(value)

        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale

        # Generate a mask tensor
        mask = torch.ones(batch_size, seq_len, seq_len).to(device)  # Assume no padding initially

        # Adjust the mask tensor to match the sequence length of the input data
        # For this example, let's assume we have variable sequence lengths in a batch
        # Here, we randomly choose sequence lengths between 1 and seq_len for each batch item
        for i in range(batch_size):
            actual_seq_len = torch.randint(1, seq_len + 1, (1,))
            mask[i, :, actual_seq_len:] = 0  # Set padding tokens to 0
            
        attention = F.softmax(energy, dim=-1)

        x = torch.matmul(self.dropout(attention), V)
        x = x.permute(0, 2, 1, 3).contiguous().view(batch_size, -1, self.input_dim)

        x = self.fc(x)
        return x

class TransformerEncoderLayer(nn.Module):
    def __init__(self, input_dim, num_heads, hidden_dim, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.multihead_attention = MultiHeadAttention(input_dim, num_heads, dropout=dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
        self.layer_norm1 = nn.LayerNorm(input_dim)
        self.layer_norm2 = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attended = self.multihead_attention(x, x, x, mask=mask)
        x = x + self.dropout(self.layer_norm1(attended))

        fed_forward = self.feed_forward(x)
        x = x + self.dropout(self.layer_norm2(fed_forward))

        return x

class TransformerEncoder(nn.Module):
    def __init__(self, input_dim, num_layers, num_heads, hidden_dim, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(input_dim, num_heads, hidden_dim, dropout=dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask=mask)
        return x

# Generate random input data and mask tensor for training
def generate_training_data(batch_size, seq_len, input_dim):
    input_data = torch.randn(batch_size, seq_len, input_dim)
    mask = torch.ones(batch_size, seq_len)  # Assume no padding initially

    for i in range(batch_size):
        actual_seq_len = torch.randint(1, seq_len + 1, (1,))
        mask[i, actual_seq_len:] = 0  # Set padding tokens to 0

    return input_data.to(device), mask.to(device)

# Generate random input data and mask tensor for testing
def generate_test_data(batch_size, seq_len, input_dim):
    return torch.randn(batch_size, seq_len, input_dim).to(device)

# Define the Transformer Encoder model and other hyperparameters
input_dim = 512
num_layers = 6
num_heads = 8
hidden_dim = 2048
dropout = 0.1
batch_size = 32
seq_len = 10

# Instantiate the Transformer Encoder model
encoder = TransformerEncoder(input_dim, num_layers, num_heads, hidden_dim, dropout=dropout).to(device)

# Define your loss function (e.g., cross-entropy loss)
criterion = nn.CrossEntropyLoss()

# Define your optimizer (e.g., Adam optimizer)
optimizer = torch.optim.Adam(encoder.parameters(), lr=0.001)

# Number of training epochs
num_epochs = 10

# Training loop
for epoch in range(num_epochs):
    # Set the model to training mode
    encoder.train()

    # Generate training data
    input_data, mask = generate_training_data(batch_size, seq_len, input_dim)

    # Clear gradients
    optimizer.zero_grad()

    # Forward pass
    output = encoder(input_data, mask=mask)

    # Generate random target labels for training (example purpose)
    target = torch.randint(0, input_dim, (batch_size, seq_len)).to(device)

    # Calculate the loss
    loss = criterion(output.view(-1, input_dim), target.view(-1))

    # Backward pass
    loss.backward()

    # Update the parameters
    optimizer.step()

    # Print training loss every epoch
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Testing loop
# Set the model to evaluation mode
encoder.eval()

# Generate test data
test_input_data = generate_test_data(batch_size, seq_len, input_dim)

# Forward pass for testing
with torch.no_grad():
    test_output = encoder(test_input_data)

print("Test output shape:", test_output.shape)  # Should be (batch_size, seq_len, input_dim)
