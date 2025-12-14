import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

data = pd.read_csv('dataset/train.csv.zip', compression='zip')
data = np.array(data)
m,n = data.shape


dataset_val = data[0:1000]
Y_val = dataset_val[:,0]
X_val = np.float32(dataset_val[:, 1:n])
X_val /= 255.0


dataset_train = data[1000:m]
Y_train = dataset_train[:,0]
X_train = np.float32(dataset_train[:, 1:n])
X_train /= 255.0


Y_val = torch.tensor(Y_val, dtype=torch.long)
X_val = torch.tensor(X_val, dtype=torch.float32)
Y_train = torch.tensor(Y_train, dtype=torch.long)
X_train = torch.tensor(X_train, dtype=torch.float32)


Val_dataset = TensorDataset(X_val,Y_val)
Train_dataset = TensorDataset(X_train,Y_train)
val_data_loader = DataLoader(Val_dataset, batch_size=10, shuffle=False)
train_data_loader = DataLoader(Train_dataset, batch_size=10, shuffle=True)



class MNISTModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(28*28,10),
            nn.ReLU(),
            nn.Linear(10,10),
        )

    def forward(self,x):
        outputs = self.model(x)

        return outputs
    

model = MNISTModel()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)


epochs = 50


def main():

    accuracies = []
    losses = []
    epoch_list = []
        
    for epoch in range(epochs):

        running_loss = 0
        total = 0
        correct = 0

        model.train()
        for Xbatch,Ybatch in train_data_loader:
            optimizer.zero_grad()
            outputs = model(Xbatch)
            loss = criterion(outputs,Ybatch)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()/Xbatch.size(0)

        epoch_training_loss = running_loss / len(train_data_loader)

        model.eval()

        with torch.no_grad():
            for Xbatch,Ybatch in train_data_loader:
                outputs = model(Xbatch)
                _,predictions = torch.max(outputs,1)
                total += Ybatch.size(0)
                correct += (predictions == Ybatch).sum().item()

        accuracy = correct/total

        print(f'Epoch {epoch}, accuracy {accuracy}, T_loss {epoch_training_loss}')

        accuracies.append(accuracy)
        losses.append(epoch_training_loss)
        epoch_list.append(epoch)

    
    fig,ax = plt.subplots(1,2,figsize = (10,4))

    ax[0].plot(epoch_list, accuracies)
    ax[0].set_title('Accuracy')
    ax[0].set_xlabel('Epoch')

    ax[1].plot(epoch_list, losses)
    ax[1].set_title('Training Loss')
    ax[1].set_xlabel('Epoch')

    plt.savefig('resources/training_fig.png')



    torch.save(model.state_dict(), 'model.pth')
    print('modelo guardado')



if __name__ == "__main__":

    main()
        

