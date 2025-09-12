Of course. Here is the English translation of the document.

# EGCN-CG

EGCN+CG

# EGCN + Softmax Architecture Explained

EGCN+CG

# EGCN + Softmax Architecture Explained

### Complete Model Pipeline

```
Input Column Features → EGCN Encoder → Node Embeddings → Softmax Classifier → Redundancy Probability
       ↓                 ↓                 ↓                  ↓                   ↓
 [features]          [embeddings]      [embeddings]         [logits]        [probabilities]
 (N, F)              GCN+GRU           (N, H)             (N, 2)            (N, 2)
```

---

### 🔧 Current Implemented Architecture

#### 1. EGCN Encoder Part

```python
# In egcn_trainer.py
self.model = EGCN(args, activation=F.relu, device=self.device)
```

**Functionality of EGCN:**

* ✅ Multi-relational graph convolution (coal, time, vessel relations)
* ✅ Dynamic weight updates via GRU
* ✅ Generates node embedding representations

#### 2. Softmax Classifier Part

```python
# In egcn_trainer.py, lines 128-133
self.classifier = nn.Sequential(
    nn.Linear(args.layer_2_feats, 16),  # Hidden Layer
    nn.ReLU(),                          # Activation Function
    nn.Dropout(0.2),                    # Prevent Overfitting
    nn.Linear(16, 2)                    # Binary Classification Output
).to(self.device)
```

**Functionality of the Softmax Classifier:**

* ✅ Maps EGCN's node embeddings to 2D logits
* ✅ Obtains a probability distribution via Softmax
* ✅ Outputs redundancy predictions

#### 3. Forward Pass Pipeline

```python
# The complete pipeline during training and inference
def forward_pass(self, features, adj_matrices):
    # 1. EGCN Encoding
    node_embeddings, _ = self.model(adj_matrices, features)
    
    # 2. Softmax Classification
    logits = self.classifier(node_embeddings)
    probabilities = F.softmax(logits, dim=1)
    predictions = torch.argmax(probabilities, dim=1)
    
    return predictions, probabilities
```

---

### 📊 Output Explanation

#### Softmax Output Format

```python
# For each column, Softmax outputs 2 probabilities:
probabilities = [
    [P(Redundant), P(Non-redundant)],     # Column 1
    [P(Redundant), P(Non-redundant)],     # Column 2
    [P(Redundant), P(Non-redundant)],     # Column 3
    ...
]

# For example:
probabilities = [
    [0.85, 0.15],  # Column 1: 85% probability of being redundant
    [0.23, 0.77],  # Column 2: 77% probability of being non-redundant  
    [0.92, 0.08],  # Column 3: 92% probability of being redundant
]
```

#### Decision Logic

```python
# Making decisions based on probability
for i, prob in enumerate(probabilities):
    redundant_prob = prob[0]      # Redundant probability
    non_redundant_prob = prob[1]  # Non-redundant probability
    
    if redundant_prob > THRESHOLD:  # e.g., 0.95
        decision = "Prune this column"
    else:
        decision = "Keep this column"
    
    print(f"Column {i}: Redundancy Prob={redundant_prob:.3f}, Decision={decision}")
```

---

### 🎯 Application in the CG Algorithm

#### Integration into the CG Algorithm

```python
# Application in CG_x_sn_rule_with_EGCN.py
def egcn_redundancy_check(column_features):
    """Perform redundancy check using EGCN+Softmax"""
    
    # 1. Prepare input data
    features = torch.tensor(column_features, dtype=torch.float32)
    adj_matrices = build_adjacency_matrices(features)
    
    # 2. EGCN forward pass
    with torch.no_grad():
        node_embeddings, _ = egcn_model(adj_matrices, features)
        
        # 3. Softmax classification
        logits = egcn_classifier(node_embeddings)
        probabilities = F.softmax(logits, dim=1)
    
    # 4. Decision based on threshold
    redundant_probs = probabilities[:, 0]  # Redundancy probabilities
    is_redundant = redundant_probs > REDUNDANCY_THRESHOLD
    
    return is_redundant, probabilities

# Usage in the main CG loop
for supplier in suppliers:
    # Generate new column
    new_column_features = solve_pricing_problem(supplier)
    
    # EGCN+Softmax redundancy check
    is_redundant, probs = egcn_redundancy_check(new_column_features)
    
    if is_redundant:
        print(f"Column pruned, redundancy probability: {probs[0]:.3f}")
        continue  # Skip this column
    else:
        print(f"Column kept, non-redundancy probability: {probs[1]:.3f}")
        add_column_to_master_problem(new_column_features)
```

---

### 🔧 Softmax during the Training Process

#### Loss Function

```python
# Training the Softmax classifier using cross-entropy loss
def train_epoch(self, train_loader):
    for batch in train_loader:
        features, adj_matrices, labels = batch
        
        # Forward pass
        node_embeddings, _ = self.model(adj_matrices, features)
        logits = self.classifier(node_embeddings)
        
        # Calculate loss (Softmax is included automatically)
        loss = F.cross_entropy(logits, labels)
        
        # Backward pass
        loss.backward()
        self.optimizer.step()
```

#### Evaluation Metrics

```python
# Calculating various metrics during evaluation
def evaluate(self, test_loader):
    all_predictions = []
    all_probabilities = []
    all_labels = []
    
    for batch in test_loader:
        features, adj_matrices, labels = batch
        
        with torch.no_grad():
            node_embeddings, _ = self.model(adj_matrices, features)
            logits = self.classifier(node_embeddings)
            
            # Softmax probabilities
            probabilities = F.softmax(logits, dim=1)
            predictions = torch.argmax(probabilities, dim=1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics like Accuracy, F1-score, etc.
    accuracy = accuracy_score(all_labels, all_predictions)
    f1 = f1_score(all_labels, all_predictions)
    
    return accuracy, f1, all_probabilities
```

---

### ⚙️ Hyperparameter Tuning

#### Classifier-related Hyperparameters

```python
# Adjustable parameters in egcn_trainer.py
classifier_config = {
    'hidden_dim': 16,        # Hidden layer dimension
    'dropout_rate': 0.2,     # Dropout rate
    'num_classes': 2,        # Number of classes (redundant/non-redundant)
}

# Training-related Hyperparameters
training_config = {
    'learning_rate': 1e-3,   # Learning rate
    'weight_decay': 1e-5,    # Weight decay
    'redundancy_threshold': 0.95,  # Redundancy judgment threshold
}
```

#### Threshold Sensitivity Analysis

```python
# Testing the impact of different thresholds
thresholds = [0.90, 0.92, 0.94, 0.95, 0.96, 0.98]

for threshold in thresholds:
    REDUNDANCY_THRESHOLD = threshold
    
    # Test on the validation set
    pruning_rate = test_pruning_performance(val_dataset, threshold)
    
    print(f"Threshold {threshold}: Pruning Rate {pruning_rate:.2%}")
```

---

### 📈 Expected Performance

#### Typical Distribution of Softmax Output

A well-trained model should produce:

**High-confidence redundant columns:**
`[0.95, 0.05]` - 95% confident it is redundant

**High-confidence non-redundant columns:**
`[0.08, 0.92]` - 92% confident it is non-redundant

**Uncertain columns:**
`[0.45, 0.55]` - Not very certain
