import torch
import time
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision.datasets as datasets


def test_model():
    start = time.time()
    # inherit Class from training script
    from train import SimpleCNN

    # Load the trained model
    device = torch.device('mps')
    model = SimpleCNN().to(device)
    model.load_state_dict(torch.load('tensor.pt'))
    model.eval()

    # Load the test data
    transformation = transforms.Compose([transforms.Resize(255), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])
    testset = datasets.ImageFolder("./testing_data/raw-img", transform=transformation)
    testloader = torch.utils.data.DataLoader(testset, batch_size = 32, shuffle = True)

    print("\n\n\nEvaluating the model now...\n\n\n")
    # Evaluate the model
    correct = 0
    total = 0
    with torch.no_grad():  # No need to track gradients for testing
        for images, labels in testloader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            failed = [['Actual', 'Predicted']]
            for i in range(labels.size(0)):
                actual = labels[i]
                pred = predicted[i]
                failed.append([actual, pred])


    accuracy = 100 * correct / total
    print(f'\n\n\nAccuracy of the model on the test images ({total} images total): {accuracy}%\n')
    for x in failed:
        print(f'{x}\n')
    end = time.time()
    print("TESTING TIME: {} minutes".format((end - start)/60))


if __name__ == "__main__":
    test_model()