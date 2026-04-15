import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import os

# Paramètres
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
NUM_CLASSES = 38
EPOCHS_INITIAL = 20
EPOCHS_FINE_TUNE = 10
DATA_DIR = 'C:\\Users\\Castel\\Desktop\\Mode_CNN'

# Étape 1 : Prétraitement - Calculer les poids des classes
class_counts = {}
for class_name in os.listdir(os.path.join(DATA_DIR, 'Train')):
    class_path = os.path.join(DATA_DIR, 'Train', class_name)
    if os.path.isdir(class_path):
        class_counts[class_name] = len(os.listdir(class_path))

total_images = sum(class_counts.values())
class_weights = {i: total_images / (NUM_CLASSES * count) for i, count in enumerate(class_counts.values())}
print("Poids des classes :", class_weights)

# Étape 2 : Prétraitement - Charger les données
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.3,
    height_shift_range=0.3,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    os.path.join(DATA_DIR, 'Train'),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True
)

val_generator = val_datagen.flow_from_directory(
    os.path.join(DATA_DIR, 'Valid'),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

test_generator = val_datagen.flow_from_directory(
    os.path.join(DATA_DIR, 'Test'),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# Vérification des shapes
images, labels = next(train_generator)
print("Shape des images batchées :", images.shape)
print("Shape des labels batchées :", labels.shape)

# Étape 3 : Construire le modèle
base_model = MobileNet(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

model = Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    Dense(512, activation='relu'),
    Dropout(0.3),
    Dense(NUM_CLASSES, activation='softmax')
])

# Étape 4 : Compiler
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Étape 5 : Entraînement - Première phase
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.00001)
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    'C:\\Users\\Castel\\Desktop\\Mode_CNN\\model_epoch_{epoch:02d}.h5',
    save_best_only=False,
    save_weights_only=False,
    period=1
)

history = model.fit(
    train_generator,
    epochs=EPOCHS_INITIAL,
    validation_data=val_generator,
    class_weight=class_weights,
    callbacks=[early_stopping, reduce_lr, checkpoint]
)

# Étape 6 : Fine-tuning
base_model.trainable = True
for layer in base_model.layers[:-20]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history_fine = model.fit(
    train_generator,
    epochs=EPOCHS_FINE_TUNE,
    validation_data=val_generator,
    class_weight=class_weights,
    callbacks=[early_stopping, reduce_lr, checkpoint]
)

# Étape 7 : Évaluation
test_loss, test_acc = model.evaluate(test_generator)
print(f"Test accuracy: {test_acc:.2f}")

y_pred = model.predict(test_generator)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = test_generator.classes
cm = confusion_matrix(y_true, y_pred_classes)

plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=test_generator.class_indices.keys(), yticklabels=test_generator.class_indices.keys())
plt.title('Matrice de confusion')
plt.xlabel('Prédictions')
plt.ylabel('Vraies étiquettes')
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.savefig('C:\\Users\\Castel\\Desktop\\Mode_CNN\\confusion_matrix.png')
plt.show()

print(classification_report(y_true, y_pred_classes, target_names=test_generator.class_indices.keys()))

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'] + history_fine.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'] + history_fine.history['val_accuracy'], label='Val Accuracy')
plt.title('Précision')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'] + history_fine.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'] + history_fine.history['val_loss'], label='Val Loss')
plt.title('Perte')
plt.legend()
plt.savefig('C:\\Users\\Castel\\Desktop\\Mode_CNN\\training_curves.png')
plt.show()

# Étape 8 : Sauvegarde
model.save('C:\\Users\\Castel\\Desktop\\Mode_CNN\\plant_disease_model_optimized.h5')

# Étape 9 : Prédiction
from tensorflow.keras.preprocessing import image

def predict_image(img_path):
    img = image.load_img(img_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    predicted_class = list(train_generator.class_indices.keys())[np.argmax(prediction)]
    confidence = np.max(prediction)
    return predicted_class, confidence

# Exemple de prédiction
img_path = 'C:\\Users\\Castel\\Desktop\\Mode_CNN\\Test\\Tomato___healthy\\0a64655c-4052-4e5e-a696-614a6133fbad___GH_HL Leaf 346.1.JPG'
predicted_class, confidence = predict_image(img_path)
print(f"Prédiction : {predicted_class} (Confiance : {confidence:.2f})")

# Afficher le résumé du modèle
model.summary()