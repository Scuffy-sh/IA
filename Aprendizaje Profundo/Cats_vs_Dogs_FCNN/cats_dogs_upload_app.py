#Importando modulos
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
from tensorflow import keras

#Para mostrar las imagenes por pantalla
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

#Para crear el modelo
from sklearn.model_selection import train_test_split

#Para la red neuronal
from keras import layers


#Definiendo la ruta del contenido que vamos a usar
DATASET_PATH = "AI_Learning\\Clasificacion_Imagenes\\PetImages"


######################################################################
##########Filtro para eliminar la imagenes que son sean JPEG##########
######################################################################

#Definiendo una función para filtrar las imagenes y quedarnos solo
#las que sean JPEG
def filter_images():
    #Iniciamos un contador
    deleted_imgs = 0
    #Accediendo a los dos directorios donde se encuentran las imagenes
    for folder_name in ("Cat","Dog"):
        #Para acceder a la ruta de cada carpeta
        folder_path = os.path.join(DATASET_PATH, folder_name)
        #Accediendo a cada imagen de cada directorio
        for image in os.listdir(folder_path):
            #Para acceder a cada imagen de la carpeta
            img_path = os.path.join(folder_path, image)
            try:
                #Abrimos la imagen 
                fobj = open(img_path, "rb")
                #Para comprobar si la imagen esta en formato JPEG
                is_jfif = tf.compat.as_bytes("JFIF") in fobj.peek(10)
            finally:
                #Para cerrar la imagen
                fobj.close()
            #Si no esta en el formato que queremos(JPEG)
            if not is_jfif:
                #Augmentamos el contador
                deleted_imgs += 1
                #Eliminamos la imagen correspondiente
                os.remove(img_path)
    
    #Mostramos por pantalla el numero de imagenes eliminadas
    print(f"Imagenes eliminadas: {deleted_imgs}")

#Llamamos a la función para que el filtrado de imagenes se ejecute
#filter_images()



##########################################################
##########Para conocer el tamaño de las imagenes##########
##########################################################


#Definiendo el tamaño de la figura donde se mostraran las imagenes
plt.figure(figsize=(10,10))

#Para acceder al directorio Dog
folder_path = os.path.join(DATASET_PATH, "Dog")
#Para recorrer las 9 primeras imagenes
for i, image in enumerate(os.listdir(folder_path)[:9]):
    #Para obtener la ruta completa de la imagen
    img_path = os.path.join(folder_path, image)
    #Leeemos la imagen
    img = mpimg.imread(img_path)
    #Organizando la figura
    #En este caso tendremos 3 filas y 3 columnas
    #subplot(nfilas,ncolumnas,index)
    ax = plt.subplot(3,3,i + 1)
    #Mostramos la imagen
    plt.imshow(img)
    #Le ponemos un titulos de la imagen con el tamaño
    plt.title(f"Tamaño: {img.shape[:2][0]} x {img.shape[:2][1]} pixeles")
    #Para que no ponga ejes en la figura
    plt.axis("off")

#Para mostrar la figura terminada
#plt.show()


##################################################################
##########Para definir un tamaño común para las imagenes##########
##########y obtener el subconjunto de entrenamiento###############
##################################################################

#Definiendo un tamaño de imagen 
image_size = (180,180)
#Tamaño de conjunto de imagenes(Tamaño de lote)
batch_size = 128

#Para invocar la funcion de obtencion de datos para el entrenamiento
train_ds = keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2, #20% de los datos forman parte del subconjunto de validacion
    subset="training",
    seed=1337,
    image_size=image_size,
    batch_size=batch_size,
)
#Para obtener el numero de lotes
print(f"Numero de lotes de entrenamiento: {len(train_ds)}")

#####################################################################
##########Para ver si el tamaño de las imagenes es correcto##########
#####################################################################

#Definiendo el tamaño de la figura donde se mostraran las imagenes
plt.figure(figsize=(10,10))

#Para recorrer nuestro conjunto de datos de entrenamiento, concretamente 1 lote(128 ejemplos)
for images, labels in train_ds.take(1):
    #Para recorrer 9 imagenes
    for i in range(9):
        #Organizando la figura
        #En este caso tendremos 3 filas y 3 columnas
        #subplot(nfilas,ncolumnas,index)
        ax = plt.subplot(3,3,i + 1)
        #Transormando las imagenes a un formato adecuado para que se puedan representar
        plt.imshow(images[i].numpy().astype("uint8"))
        #Le ponemos un titulos de la imagen con el tamaño
        plt.title(f"Tamaño: {images[i].shape[0]} x {images[i].shape[1]} pixeles")
        #Para que no ponga ejes en la figura
        plt.axis("off")

#Para mostrar la figura terminada
#plt.show()


####################################################################
##########Obtenemos el subconjunto de validacion y pruebas##########
####################################################################

#Para invocar la funcion de obtencion de datos para el entrenamiento
temp_val_ds = keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2, #20% de los datos forman parte del subconjunto de validacion
    subset="validation",
    seed=1337,
    image_size=image_size,
    batch_size=batch_size,
)
#Para mostrar el numero de lotes de validacion
print(f"Numero de lotes de validación: {len(temp_val_ds)}")

#De este conjunto de lotes nos quedaremos con la mitad para validacion
#y la otra mitad para pruebas
val_size = int(0.5 * len(temp_val_ds))
val_ds = temp_val_ds.take(val_size)
test_ds = temp_val_ds.skip(val_size)
print(f"La cantidad de lotes para la validación són: {len(val_ds)}")
print(f"La cantidad de lotes para las pruebas són: {len(test_ds)}")



#####################################################################################
##########Otra forma de obtenencion del subconjunto de validacion y pruebas##########
#####################################################################################


# train_test_split no puede trabajar con objetos Dataset de Tensorflow
# Esto supone un incremento del consumo de memoria RAM
#Por eso lo convertiremos en una lista
val_ds_sk = list(temp_val_ds)

#Para dividir del conjunto de validación en validación y pruebas
val_ds_sk, test_ds_sk = train_test_split(
    val_ds_sk,
    test_size=0.5,  #Porcentaje para prueba
    random_state=42,    #Semilla para reproducibilidad
)
print(f"La cantidad de lotes para la validación són: {len(val_ds_sk)}")
print(f"La cantidad de lotes para las pruebas són: {len(test_ds_sk)}")



######################################################################
##########Definiendo La arquitectura de nuestra red neuronal##########
######################################################################


#Definiendo la dimension de los datos de entrada (pixles x pixeles, rgb)
input_shape = (180,180,3)

#Definiendo la red neuronal, en este caso sera sequencial
fcnn_model = keras.Sequential()

#Definiendo las diferentes capas
#Entrada de la red neuronal
fcnn_model.add(layers.Input(shape=input_shape))

#Escalando las imagenes
fcnn_model.add(layers.Rescaling(1.0 / 255))

#Estirar o aplanar las imagenes para la primera capa densa
fcnn_model.add(layers.Flatten())

#Capa 1 (Numero de neuronas, función matemàtica)
fcnn_model.add(layers.Dense(384, activation='relu'))

#Capa 2 (Numero de neuronas, función matemàtica)
fcnn_model.add(layers.Dense(256, activation='relu'))

#Capa 3 (Numero de neuronas, función matemàtica)
fcnn_model.add(layers.Dense(128, activation='relu'))

#Capa 4 - Output Layer. Terminamos con una neurona
#ya que lo que queremos es una clasificación binaria de los datos(gato o perro)
#(Numero de neuronas, función matemàtica)
#Si fuesen dos o mas neuronas/clases deberiamos cambiar
#la función de activación por 'softmax'
fcnn_model.add(layers.Dense(1, activation='sigmoid'))


#Para ver un resumen de lo que hemos hecho previamente
fcnn_model.summary()



#####################################################
##########Configurando nuestra red neuronal##########
#####################################################


#Compilamos el primer modelo de FCNN
#loss --> Classificacion de Error binaria. Si no fuese binaria habria que poner 'categorical_crossentropy'
#optimizer --> Adam de las más utilizadas. Adam(Learning Rate)
#metrics --> Función utilizada para la red neuronal
fcnn_model.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(1e-3), metrics=['accuracy'])



###################################################
##########Entrenando nuestra red neuronal##########
###################################################


#Proceso de entrenamiento
#epochs --> Las vueltas que da sobre los datos de entrenamiento
history = fcnn_model.fit(train_ds, epochs=10, validation_data=val_ds)

