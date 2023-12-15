import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from tqdm import tqdm

# Defining the CNN Model
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.act1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.act2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 56 * 56, 256)
        self.act3 = nn.ReLU()
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.pool1(self.act1(self.conv1(x)))
        x = self.pool2(self.act2(self.conv2(x)))
        x = x.view(x.size(0), -1)  # Automatically adapt to the input size
        x = self.act3(self.fc1(x))
        x = self.fc2(x)
        return x

def get_device():
    if torch.backends.mps.is_available():  # Hypothetical check for MPS
        return torch.device("mps")  # Hypothetical MPS device
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")



def train_model():
    start = time.time()
    print("\nTRAINING MODEL\n")

    device = get_device()
    print(f"\n Using device: {device}")

    # Instantiate the model, loss function, and optimizer
    model = SimpleCNN().to(device)
    try:
        model.load_state_dict(torch.load('tensor.pt'))
        print("\n\n Tensors Loaded... \n\n")
    except:
        print("\n\n No Existing Tensors... Generating Weights...\n\n")
        pass
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Configure data preprocessing
    transformation = transforms.Compose([transforms.Resize(255), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])

    # Load training data
    trainset = datasets.ImageFolder("./training_data/raw-img", transform=transformation)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size = 32, shuffle = True)

    # Training Loop
    num_epochs = 10
    for epoch in range(num_epochs):
        # Wrap your data loader with the tqdm class for a progress bar
        with tqdm(total=len(trainloader), desc=f'Epoch {epoch+1}/{num_epochs}') as pbar:
            for i, (images, labels) in enumerate(trainloader):
                # Move tensors to the configured device
                images = images.to(device)
                labels = labels.to(device)

                # Forward pass
                outputs = model(images)
                loss = criterion(outputs, labels)

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                pbar.update(1)
                pbar.set_postfix(loss=loss.item())

    end = time.time()
    print("\n\nTraining complete!\nTIME ELAPSED: {} minutes".format((end-start) / 60))

    # Save the trained model
    torch.save(model.state_dict(), 'tensor.pt')

if __name__ == "__main__":
    train_model()