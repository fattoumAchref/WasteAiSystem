import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing import image

# 1. Vérification de l'environnement
print(" Vérification de l'environnement...")
print(f"TensorFlow version: {tf.__version__}")
print("GPU disponible:", tf.config.list_physical_devices('GPU'))

# 2. Configuration
MODEL_PATH = 'efficient1waste15.keras'
IMAGE_PATH = 'alum.jpg'
CLASS_NAMES =['aerosol_cans', 'aluminum_food_cans', 'aluminum_soda_cans', 'cardboard_boxes', 'clothing', 'food_waste', 'glass_beverage_bottles', 'glass_cosmetic_containers', 'office_paper', 'paper_cups', 'plastic_detergent_bottles', 'plastic_shopping_bags', 'plastic_soda_bottles', 'plastic_straws', 'plastic_water_bottles']



# 3. Chargement spécial du modèle avec contournement GPU si nécessaire
try:
    # Option 1: Essai avec GPU
    print("\n Tentative de chargement avec GPU...")
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)

    # Option 2: Si échec, forcer le CPU
    try:
        with tf.device('/CPU:0'):
            model = tf.keras.models.load_model(MODEL_PATH, compile=False, safe_mode=False)
        print(" Modèle chargé sur CPU")
    except Exception as e:
        print(f"Échec du chargement: {str(e)}")
        raise

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

    # 4. Préparation de l'image (méthode robuste)
    def prepare_image(img_path):
        img = image.load_img(img_path, target_size=(256, 256))
        img_array = image.img_to_array(img)

        # Essai avec différentes méthodes de normalisation
        for method in ['efficientnet', 'standard']:
            try:
                if method == 'efficientnet':
                    from keras.applications.efficientnet import preprocess_input
                    img_processed = preprocess_input(img_array.copy())
                else:
                    img_processed = img_array / 255.0

                return np.expand_dims(img_processed, axis=0)
            except:
                continue
        return np.expand_dims(img_array / 255.0, axis=0)

    img_ready = prepare_image(IMAGE_PATH)

    # 5. Prédiction
    print("\n Prédiction en cours...")
    with tf.device('/CPU:0'):  # Forcer le CPU pour la prédiction
        preds = model.predict(img_ready)[0]

    # 6. Affichage des résultats
    pred_class = CLASS_NAMES[np.argmax(preds)]
    confidence = np.max(preds)

    plt.figure(figsize=(10, 6))
    plt.imshow(image.load_img(IMAGE_PATH))
    plt.axis('off')
    plt.title(f"Prédiction: {pred_class}\nConfiance: {confidence:.1%}",
              color='green' if confidence > 0.7 else 'orange',
              fontsize=14, pad=20)
    plt.show()



except Exception as e:
    print(f"\n ERREUR CRITIQUE: {str(e)}")
    print("\nPROCÉDURE DE DÉPANNAGE COMPLÈTE:")
    print("1. Redémarrez le runtime (Runtime → Redémarrer le runtime)")
    print("2. Exécutez ces commandes dans une nouvelle cellule:")
    print("   !pip uninstall -y tensorflow keras")
    print("   !pip install tensorflow==2.12.0")
    print("3. Vérifiez que:")
    print("   - Votre modèle est bien nommé 'wasteclasses.keras'")
    print("   - Votre image est dans /content")
    print("4. Réessayez avec cette cellule")