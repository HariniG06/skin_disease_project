import os, shutil, numpy as np, pandas as pd
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ── Config ────────────────────────────────────────────────
IMG_SIZE   = 224
BATCH_SIZE = 16
EPOCHS     = 20
SEED       = 42
ARCHIVE    = "archive"
META_CSV   = f"{ARCHIVE}/HAM10000_metadata.csv"
IMG_DIRS   = [f"{ARCHIVE}/HAM10000_images_part_1",
              f"{ARCHIVE}/HAM10000_images_part_2"]
DATA_DIR   = "data"
MODEL_PATH = "skin_disease_model.h5"

# ── Step 1: Organize images into data/train & data/test ──
def organize_data():
    if os.path.exists(f"{DATA_DIR}/train") and \
       len(os.listdir(f"{DATA_DIR}/train")) > 0:
        print("✔️ Data already organized")
        return

    df = pd.read_csv(META_CSV)
    label_map = {
        'nv':'Melanocytic_Nevi','mel':'Melanoma',
        'bkl':'Benign_Keratosis','bcc':'Basal_Cell_Carcinoma',
        'akiec':'Actinic_Keratosis','vasc':'Vascular',
        'df':'Dermatofibroma'
    }
    df['label'] = df['dx'].map(label_map)

    # Build image path lookup
    img_lookup = {}
    for d in IMG_DIRS:
        for f in os.listdir(d):
            img_lookup[f.replace('.jpg','')] = os.path.join(d, f)

    train_df, test_df = train_test_split(
        df, test_size=0.15, stratify=df['label'], random_state=SEED)

    for split, split_df in [('train', train_df), ('test', test_df)]:
        for _, row in split_df.iterrows():
            img_id = row['image_id']
            label  = row['label']
            dst_dir = f"{DATA_DIR}/{split}/{label}"
            os.makedirs(dst_dir, exist_ok=True)
            src = img_lookup.get(img_id)
            if src:
                shutil.copy(src, f"{dst_dir}/{img_id}.jpg")

    print(f"✔️ Train: {len(train_df)}  Test: {len(test_df)}")

# ── Step 2: Data Generators ───────────────────────────────
def build_generators():
    train_aug = ImageDataGenerator(
        rescale=1./255, rotation_range=30,
        width_shift_range=0.15, height_shift_range=0.15,
        zoom_range=0.2, horizontal_flip=True,
        vertical_flip=True, validation_split=0.15)
    test_aug = ImageDataGenerator(rescale=1./255)

    train_gen = train_aug.flow_from_directory(
        f"{DATA_DIR}/train", target_size=(IMG_SIZE,IMG_SIZE),
        batch_size=BATCH_SIZE, class_mode="categorical",
        subset="training", seed=SEED)
    val_gen = train_aug.flow_from_directory(
        f"{DATA_DIR}/train", target_size=(IMG_SIZE,IMG_SIZE),
        batch_size=BATCH_SIZE, class_mode="categorical",
        subset="validation", seed=SEED)
    test_gen = test_aug.flow_from_directory(
        f"{DATA_DIR}/test", target_size=(IMG_SIZE,IMG_SIZE),
        batch_size=BATCH_SIZE, class_mode="categorical",
        shuffle=False)

    print(f"✔️ Classes: {list(train_gen.class_indices.keys())}")
    print(f"✔️ Train={train_gen.samples} Val={val_gen.samples} Test={test_gen.samples}")
    return train_gen, val_gen, test_gen

# ── Step 3: Build Model ───────────────────────────────────
def build_model(num_classes):
    base = EfficientNetB0(weights="imagenet", include_top=False,
                          input_shape=(IMG_SIZE,IMG_SIZE,3))
    base.trainable = False
    inputs = tf.keras.Input(shape=(IMG_SIZE,IMG_SIZE,3))
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(num_classes, activation="softmax")(x)
    model = models.Model(inputs, out)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss="categorical_crossentropy", metrics=["accuracy"])
    return model, base

# ── Step 4: Train ─────────────────────────────────────────
def train(model, base, train_gen, val_gen):
    cbs = [
        EarlyStopping(patience=5, restore_best_weights=True, monitor="val_accuracy"),
        ReduceLROnPlateau(factor=0.5, patience=3, monitor="val_loss", verbose=1),
        ModelCheckpoint(MODEL_PATH, save_best_only=True, monitor="val_accuracy", verbose=1),
    ]
    print("\n── Phase 1: Head only ──")
    h1 = model.fit(train_gen, validation_data=val_gen, epochs=10, callbacks=cbs)

    print("\n── Phase 2: Fine-tuning ──")
    base.trainable = True
    for layer in base.layers[:-30]:
        layer.trainable = False
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
                  loss="categorical_crossentropy", metrics=["accuracy"])
    h2 = model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS, callbacks=cbs)
    return h1, h2

# ── Step 5: Evaluate ──────────────────────────────────────
def evaluate(model, test_gen):
    loss, acc = model.evaluate(test_gen)
    print(f"Accuracy: {acc*100:.2f}%  Loss: {loss:.4f}")
    preds  = model.predict(test_gen)
    y_pred = np.argmax(preds, axis=1)
    y_true = test_gen.classes
    labels = list(test_gen.class_indices.keys())
    print(classification_report(y_true, y_pred, target_names=labels))
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10,8))
    sns.heatmap(cm, annot=True, fmt="d",
                xticklabels=labels, yticklabels=labels, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150)
    print("✔️ confusion_matrix.png saved")

# ── Step 6: Plot History ──────────────────────────────────
def plot_history(h1, h2):
    acc  = h1.history["accuracy"]     + h2.history["accuracy"]
    val  = h1.history["val_accuracy"] + h2.history["val_accuracy"]
    loss = h1.history["loss"]         + h2.history["loss"]
    vloss= h1.history["val_loss"]     + h2.history["val_loss"]
    fig, (ax1,ax2) = plt.subplots(1,2,figsize=(14,5))
    ax1.plot(acc,label="Train"); ax1.plot(val,label="Val")
    ax1.set_title("Accuracy"); ax1.legend()
    ax2.plot(loss,label="Train"); ax2.plot(vloss,label="Val")
    ax2.set_title("Loss"); ax2.legend()
    plt.tight_layout()
    plt.savefig("training_history.png", dpi=150)
    print("✔️ training_history.png saved")

# ── Main ──────────────────────────────────────────────────
if __name__ == "__main__":
    organize_data()
    train_gen, val_gen, test_gen = build_generators()
    import json
    with open("classes.json","w") as f:
        json.dump(list(train_gen.class_indices.keys()), f)
    num_classes = len(train_gen.class_indices)
    model, base = build_model(num_classes)
    h1, h2 = train(model, base, train_gen, val_gen)
    plot_history(h1, h2)
    evaluate(model, test_gen)
    print(f"✔️ Model saved → {MODEL_PATH}")