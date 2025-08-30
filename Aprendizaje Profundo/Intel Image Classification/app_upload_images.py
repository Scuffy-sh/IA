# Importar librerias
import streamlit as st
import torch
from torchvision import models, transforms
from PIL import Image # Para abrir imagenes
import numpy as np
import pandas as pd

# Cargar el modelo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Cargar el checkpoint
checkpoint = torch.load("D:\Hacking\Python\AI_Learning\Aprendizaje_Profundo\Intel_Image_Class_PyTorch_CNN\EfficientNetB0_phase2.pth", map_location=device)

# Recuperar información del checkpoint
num_classes = checkpoint['num_classes']

# Reconstruir el modelo
def build_model(num_classes):
    model = models.efficientnet_b0(weights=None) # Cargar sin pesos
    in_features = model.classifier[-1].in_features # Obtener el número de características de entrada
    model.classifier = torch.nn.Sequential(
        torch.nn.Linear(in_features, 256),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.4),
        torch.nn.Linear(256, num_classes)
    )
    return model


model = build_model(num_classes).to(device)
model.load_state_dict(checkpoint['state_dict'])
model.eval()


# Mapeo idx-->clase
idx_to_class = {
    0: "buildings",
    1: "forest",
    2: "glacier",
    3: "mountain",
    4: "sea",
    5: "street"
}

# Definir transformaciones

imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]
img_size = 224

eval_tfms = transforms.Compose([
    transforms.Resize(int(img_size * 1.14)),
    transforms.CenterCrop(img_size),
    transforms.ToTensor(),
    transforms.Normalize(imagenet_mean, imagenet_std)
])

# Interfaz
st.title("Clasificación de Imágenes con EfficientNetB0 - Intel Image Classification")
st.write("Sube una imagen y el modelo clasificará la imagen según la categoría.")

uploaded_file = st.file_uploader("Selecciona una imagen", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Imagen subida", use_container_width=True)

    # Preprocesar la imagen
    img_tensor = eval_tfms(image).unsqueeze(0).to(device)

    # Realizar la predicción
    with st.spinner('Realizando la predicción...'):
        with torch.no_grad():
            output = model(img_tensor)  # Realizar la predicción
            probs = torch.nn.functional.softmax(output, dim=1) # Obtener probabilidades
            pred_idx = torch.argmax(probs, dim=1).item() # Obtener índice de la clase predicha
            pred_class = idx_to_class[pred_idx] # Obtener nombre de la clase predicha
            confidence = probs[0][pred_idx].item() # Obtener confianza de la predicción

    # Mostrar resultados
    st.markdown(f"### Predicción: {pred_class.capitalize()}")
    st.markdown(f"### Confianza: {confidence * 100:.2f}%")

    # Mostrar las 3 principales predicciones
    top_probs, top_idxs = torch.topk(probs, 3)
    st.subheader("Top 3 Predicciones:")
    top3_df = pd.DataFrame({
        "Clase": [idx_to_class[idx.item()].capitalize() for idx in top_idxs[0]],
        "Confianza": [f"{prob.item() * 100:.2f} %" for prob in top_probs[0]]
    })
    st.table(top3_df)