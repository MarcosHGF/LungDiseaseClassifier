import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import kagglehub
import os
import matplotlib.pyplot as plt
from flask import Flask, request, render_template

# Starting Flask

app = Flask(__name__)

# Download latest version
#kagglehub.dataset_download("rifatulmajumder23/combined-unknown-pneumonia-and-tuberculosis")

# Transform and normalization
transform = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.497, 0.501, 0.499], std=[0.248, 0.248, 0.252])
])

# Load Data
train_data = datasets.ImageFolder(os.path.join('data/test'), transform=transform)
val_data = datasets.ImageFolder(os.path.join('data/val'), transform=transform)
test_data = datasets.ImageFolder(os.path.join('data/test'), transform=transform)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32, shuffle=False)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

# Used to get the mean and the std of the data and normalize

'''
# Calcular a média e desvio padrão
mean = torch.zeros(3)
std = torch.zeros(3)
total_images = 0

for images, _ in train_loader:
    batch_samples = images.size(0)  # Número de imagens no batch
    total_images += batch_samples
    mean += images.mean([0, 2, 3]) * batch_samples
    std += images.std([0, 2, 3]) * batch_samples

mean /= total_images
std /= total_images

print(f"Média: {mean}")
print(f"Desvio padrão: {std}")

# Substituir as transformações com normalização personalizada
transform = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[mean[0].item(), mean[1].item(), mean[2].item()],
                         std=[std[0].item(), std[1].item(), std[2].item()])
])

# Recarregar os dados com as novas transformações
train_data = datasets.ImageFolder(os.path.join('data/train'), transform=transform)
val_data = datasets.ImageFolder(os.path.join('data/val'), transform=transform)
test_data = datasets.ImageFolder(os.path.join('data/test'), transform=transform)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32, shuffle=False)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

'''

# Simple CNN model
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 18 * 18, 512)
        self.fc2 = nn.Linear(512, 4)  # classes: NORMAL, PNEUMONIA, TUBERCULOSIS, NON-XRAY

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(-1, 128 * 18 * 18)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# initialize model, loss function e optimizer
model = CNNModel()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Creating list to later plot the loss and accuracy

train_losses = []
train_accuracies = []

# Training

num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    train_losses.append(running_loss / len(train_loader))
    train_accuracies.append(accuracy)

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}, Accuracy: {accuracy:.2f}%')

# Plot Loss and Accuracy
fig, ax1 = plt.subplots(figsize=(10, 6))

# Ploting Loss
ax1.plot(range(num_epochs), train_losses, 'b-', label='Loss', color='red')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss', color='red')
ax1.tick_params(axis='y', labelcolor='red')

# Creating y axis for Accuracy
ax2 = ax1.twinx()
ax2.plot(range(num_epochs), train_accuracies, 'g-', label='Accuracy', color='green')
ax2.set_ylabel('Accuracy (%)', color='green')
ax2.tick_params(axis='y', labelcolor='green')

# title and show
plt.title('Loss and Accuracy over Epochs')
plt.tight_layout()
#plt.show()

# Aval

model.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in val_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Validation Accuracy: {100 * correct / total:.2f}%')

# Predicting with a input

from PIL import Image

def predict_image(image_path, model, transform):
    # Load Image
    image = Image.open(image_path).convert('RGB')
    
    # Apply transforms
    image = transform(image).unsqueeze(0)
    
    # Aval mode
    model.eval()
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
    
    # Get predicted class
    classes = ['NORMAL', 'PNEUMONIA', 'TUBERCULOSIS', 'NON-XRAY']
    return classes[predicted.item()]

# Testing
image_path = '13.png'  # Put your image to test
prediction = predict_image(image_path, model, transform)
print(f'Predicted Class: {prediction}')

# Graphic visualization

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return 'No file uploaded!', 400
        file = request.files['file']
        if file.filename == '':
            return 'No selected file!', 400
        
        # Save image
        file_path = os.path.join('static','uploads', file.filename)
        file.save(file_path)
        #print(file_path)

        # Make predictions
        prediction = predict_image(file_path, model, transform)

        # Pass to Front
        return render_template('index.html', image_url=file_path, prediction=prediction)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)