#Para importar la libreria donde podremos descargar los datasets
from datasets import load_dataset

#Para cargar un dataset concreto
dataset = load_dataset("hf-internal-testing/cats_vs_dogs_sample")
print(dataset)