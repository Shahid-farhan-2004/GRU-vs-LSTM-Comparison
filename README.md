# LSTM vs GRU Sequence Classification using PyTorch

A simple PyTorch implementation demonstrating **sequence classification** using both **LSTM (Long Short-Term Memory)** and **GRU (Gated Recurrent Unit)**. The project compares the two recurrent neural network architectures on a toy binary classification dataset.

---

## Features

- Custom PyTorch `Dataset`
- Variable-length sequence handling using `pad_sequence()`
- Embedding layer
- LSTM model
- GRU model
- Shared model architecture
- Binary sequence classification
- CrossEntropyLoss
- Adam optimizer

---

## Dataset

### Input Sequences

```python
sequences = [
    [1, 2, 3],
    [4, 5],
    [1, 1, 1],
    [5, 4, 3],
    [2, 2],
    [0, 1, 0]
]
```

### Labels

```python
labels = [1, 0, 1, 0, 1, 0]
```

Each sequence belongs to one of two classes:

- **0**
- **1**

---

## Model Architecture

```
Input Sequence
      │
      ▼
Embedding Layer
(Embedding Size = 16)
      │
      ▼
LSTM / GRU
(Hidden Size = 32)
      │
      ▼
Last Hidden State
(out[:, -1, :])
      │
      ▼
Fully Connected Layer
      │
      ▼
2 Output Classes
```

---

## Project Structure

```
.
├── rnn_classifier.py
└── README.md
```

---

## Requirements

- Python 3.x
- PyTorch

Install PyTorch:

```bash
pip install torch
```

---

## Run

```bash
python rnn_classifier.py
```

---

## Handling Variable-Length Sequences

The sequences have different lengths.

Example:

```
[1, 2, 3]
[4, 5]
```

The custom `collate_fn` pads shorter sequences so they can be batched together.

```python
def collate_fn(batch):
    xs, ys = zip(*batch)

    xs = pad_sequence(
        xs,
        batch_first=True,
        padding_value=0
    )

    ys = torch.tensor(ys)

    return xs, ys
```

Example after padding:

```
[
 [1,2,3],
 [4,5,0]
]
```

---

## Model

### Embedding Layer

```python
self.embedding = nn.Embedding(6, 16)
```

Converts integer tokens into 16-dimensional vectors.

---

### LSTM

```python
self.rnn = nn.LSTM(
    input_size=16,
    hidden_size=32,
    batch_first=True
)
```

---

### GRU

```python
self.rnn = nn.GRU(
    input_size=16,
    hidden_size=32,
    batch_first=True
)
```

---

### Classification Layer

```python
self.fc = nn.Linear(32, 2)
```

Produces scores for the two output classes.

---

## Forward Pass

```python
x = self.embedding(x)

out, _ = self.rnn(x)

out = out[:, -1, :]

return self.fc(out)
```

The last hidden state is used as the representation of the entire sequence.

---

## Tensor Shapes

### Input

```
(batch_size, sequence_length)

Example:

(2,3)
```

---

### After Embedding

```
(batch_size, sequence_length, 16)

(2,3,16)
```

---

### After LSTM / GRU

```
(batch_size, sequence_length, 32)

(2,3,32)
```

---

### Selecting the Last Hidden State

```python
out = out[:, -1, :]
```

Shape:

```
(batch_size, 32)

(2,32)
```

---

### Final Output

```
(batch_size, num_classes)

(2,2)
```

Each row contains the prediction scores for:

- Class 0
- Class 1

---

## Loss Function

```python
criterion = nn.CrossEntropyLoss()
```

Since there are two output classes, `CrossEntropyLoss` is used.

---

## Optimizer

```python
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)
```

---

## Training Process

1. Load sequences.
2. Pad sequences within each batch.
3. Convert tokens into embeddings.
4. Pass embeddings through an LSTM or GRU.
5. Extract the last hidden state.
6. Predict the class.
7. Compute CrossEntropyLoss.
8. Update model weights using Adam.

---

## Example Output

```
Training lstm

Epoch 1 Average Loss: 0.71
Epoch 2 Average Loss: 0.64
Epoch 3 Average Loss: 0.57
Epoch 4 Average Loss: 0.49
Epoch 5 Average Loss: 0.42

Training gru

Epoch 1 Average Loss: 0.69
Epoch 2 Average Loss: 0.60
Epoch 3 Average Loss: 0.52
Epoch 4 Average Loss: 0.46
Epoch 5 Average Loss: 0.39
```

*Loss values may vary due to random initialization.*

---

## LSTM vs GRU

| Feature | LSTM | GRU |
|---------|------|-----|
| Gates | Forget, Input, Output | Reset, Update |
| Cell State | Yes | No |
| Hidden State | Separate | Single |
| Parameters | More | Fewer |
| Training Speed | Slower | Faster |
| Memory Usage | Higher | Lower |

---

## Future Improvements

- Use Bidirectional LSTM/GRU
- Add Dropout
- Use Packed Sequences (`pack_padded_sequence`)
- Train on a larger NLP dataset
- Replace RNNs with Transformer models

---

## Technologies Used

- Python
- PyTorch
- Dataset
- DataLoader
- pad_sequence
- Embedding Layer
- LSTM
- GRU
- CrossEntropyLoss
- Adam Optimizer

---

## License

This project is released under the MIT License.
