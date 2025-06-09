import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tkinter import *
from PIL import Image, ImageTk
import os

# Verificar y cargar datos
if not os.path.exists("dataset"):
    print("Error: No se encontró la carpeta dataset. Asegúrate de que las imágenes están organizadas correctamente.")
    exit()

dataset = tf.keras.preprocessing.image_dataset_from_directory(
    "dataset", batch_size=32, image_size=(128, 128), label_mode="int"
)

# Normalización de datos
normalization_layer = layers.Rescaling(1./255)
dataset = dataset.map(lambda x, y: (normalization_layer(x), y))

# Definir modelo CNN
modelo = keras.Sequential([
    layers.Conv2D(32, (3,3), activation="relu", input_shape=(128,128,3)),
    layers.MaxPooling2D(),
    layers.Conv2D(64, (3,3), activation="relu"),
    layers.MaxPooling2D(),
    layers.Conv2D(128, (3,3), activation="relu"),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(5, activation="softmax")  # 5 clases de gestos
])

# Compilar modelo
modelo.compile(optimizer="adam",
               loss="sparse_categorical_crossentropy",
               metrics=["accuracy"])

# Entrenar modelo
modelo.fit(dataset, epochs=15)

# Guardar modelo entrenado
modelo.save("modelo_gestos.h5")

# Cargar modelo entrenado
model = tf.keras.models.load_model('modelo_gestos.h5')

# Categorías de los gestos
categorias = ['Feliz', 'Sorprendido', 'Enojado', 'Triste', 'Neutral']

# Detector Viola-Jones
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Configuración inicial de Tkinter
root = Tk()
root.title("Reconocimiento de Gestos Faciales")
root.geometry("800x600")

# Video desde webcam
cap = cv2.VideoCapture(0)

# Etiqueta para video
label_video = Label(root)
label_video.pack()

# Etiqueta para el resultado del gesto
label_resultado = Label(root, text="Detectando gesto...", font=("Arial", 20))
label_resultado.pack()

# Función para detectar gestos y encuadrar con verde
def detectar_gestos(frame):
    gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rostros = face_cascade.detectMultiScale(gris, 1.1, 4)

    for (x, y, w, h) in rostros:
        rostro = frame[y:y+h, x:x+w]
        rostro_resized = cv2.resize(rostro, (128, 128))
        rostro_norm = np.expand_dims(rostro_resized / 255.0, axis=0)

        prediccion = model.predict(rostro_norm)
        clase = categorias[np.argmax(prediccion)]

        # Dibujar el rectángulo verde y la etiqueta del gesto
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, clase, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        return clase
    return "Sin rostro"

# Función para mostrar video en tiempo real
def mostrar_frame():
    ret, frame = cap.read()
    gesto_detectado = detectar_gestos(frame)
    label_resultado.config(text=f"Gesto detectado: {gesto_detectado}")

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img_rgb)
    imgtk = ImageTk.PhotoImage(image=img)

    label_video.imgtk = imgtk
    label_video.configure(image=imgtk)
    label_video.after(10, mostrar_frame)

# Ejecutar la función
mostrar_frame()

# Iniciar interfaz Tkinter
root.mainloop()

# Al cerrar la ventana, liberar recursos
cap.release()
