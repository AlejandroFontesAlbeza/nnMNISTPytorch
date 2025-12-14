from train import X_val,Y_val,MNISTModel
import torch
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd


dataset = pd.read_csv('dataset/test.csv.zip', compression='zip')
data = np.array(dataset)


idx = 105
image = data[idx]        # shape = (784,)
image_tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0)  # añadir batch dim → (1, 784)
image_tensor /= 255.0

model = MNISTModel()

model.load_state_dict(torch.load("model.pth"))
model.eval()

def main():
    img = image_tensor

    img_plt = img.reshape(28,28) * 255.0

    # Inferencia
    with torch.no_grad():
        outputs = model(img)
        predicted_label = torch.argmax(outputs, dim=1).item()

    # Mostrar la imagen
    plt.gray()
    plt.imshow(img_plt)
    plt.title(f"Predicted: {predicted_label}")
    plt.savefig("resources/label_predicted")
    print('label predicted')


if __name__ == "__main__":
    main()