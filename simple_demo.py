# simple_demo.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os


class SimpleDataset(Dataset):
    def __init__(self, data_path, split):
        self.samples = []
        print(f"load{split}data...")
        
        for i in range(10):
            self.samples.append({
                "text": f"Sample text {i}",
                "aspects": [f"aspect{j}" for j in range(2)],
                "opinions": [f"opinion{j}" for j in range(2)],
                "sentiment": ["positive", "negative"]
            })
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]


class SimpleASTEModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(100, 64)
        self.lstm = nn.LSTM(64, 64, batch_first=True)
        self.aspect_head = nn.Linear(64, 3)
        self.opinion_head = nn.Linear(64, 3)
        self.sentiment_head = nn.Linear(64, 3)
    
    def forward(self, x):
        
        indices = torch.tensor([ord(c) % 100 for c in x["text"]])
        embedded = self.embedding(indices).unsqueeze(0)
        lstm_out, _ = self.lstm(embedded)
        pooled = torch.mean(lstm_out, dim=1)
        
        aspects = self.aspect_head(pooled)
        opinions = self.opinion_head(pooled)
        sentiments = self.sentiment_head(pooled)
        
        return {"aspects": aspects, "opinions": opinions, "sentiments": sentiments}


def train_simple_model():
    print("load data...")
    train_data = SimpleDataset("dataset/data", "train")
    train_loader = DataLoader(train_data, batch_size=2)
    
    print("load data...")
    model = SimpleASTEModel()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    print("begin training...")
    for epoch in range(2):
        for batch in train_loader:
            optimizer.zero_grad()
            
            
            for sample in batch:
                outputs = model(sample)
                
                
                target = torch.tensor([0])  
                loss = criterion(outputs["aspects"], target) + criterion(outputs["opinions"], target) + criterion(outputs["sentiments"], target)
                
                loss.backward()
            
            optimizer.step()
        
        print(f"fininsh epoch{epoch+1}/2")
    
    print("savemodel...")
    os.makedirs("demo_results", exist_ok=True)
    torch.save(model.state_dict(), "demo_results/model.pt")
    
    print("demo complete!")

if __name__ == "__main__":
    train_simple_model()