import streamlit as st
import torch
import torch.nn as nn
import os
from torchvision import transforms
from PIL import Image


class BeltClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv1_bn = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv2_bn = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3_bn = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv4_bn = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv5_bn = nn.BatchNorm2d(256)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(256 * 7 * 7, 512)
        self.fc1_bn = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 128)
        self.fc2_bn = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 64)
        self.fc3_bn = nn.BatchNorm1d(64)
        self.fc4 = nn.Linear(64, 8)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1_bn(self.conv1(x))))
        x = self.pool(nn.functional.relu(self.conv2_bn(self.conv2(x))))
        x = self.pool(nn.functional.relu(self.conv3_bn(self.conv3(x))))
        x = self.pool(nn.functional.relu(self.conv4_bn(self.conv4(x))))
        x = self.pool(nn.functional.relu(self.conv5_bn(self.conv5(x))))
        x = x.view(-1, 256*7*7)
        x = self.dropout(nn.functional.relu(self.fc1_bn(self.fc1(x))))
        x = self.dropout(nn.functional.relu(self.fc2_bn(self.fc2(x))))
        x = self.dropout(nn.functional.relu(self.fc3_bn(self.fc3(x))))
        x = self.fc4(x)
        return x


def load_model_and_transform():
    model = BeltClassifier()
    model_path = os.path.join(
        os.getcwd(), 'pytorch/belt/belt_classifier.pt')
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
    ])
    return model, transform


model, transform = load_model_and_transform()

class_names = ['Braided', 'Canvas', 'Chain',
               'Leather', 'Metal', 'Misc', 'Punk', 'Wide']

st.title('Belt Image Classification')

uploaded_image = st.sidebar.file_uploader(
    "Upload an image...", type=["jpg", "png", "jpeg"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    st.image(image, caption='Uploaded Image.', width=500)
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        prediction = model(image)
    prediction = (prediction >= 0.5).squeeze().numpy()
    st.subheader("Predicted Material/Attributes :")
    predicted_labels = [class_names[i] for i, is_label_present in enumerate(
        prediction) if is_label_present]
    for label in predicted_labels:
        st.write(label)
