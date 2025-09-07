# finetune.py
import keras, numpy as np, pandas as pd, pathlib
from keras import layers, models
from PIL import Image, ImageOps
import argparse, os

# --- HYPERPARAMS ---
H, W = 50, 200
VOCAB = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
T = 5  # captcha length
MAP = {ch:i for i,ch in enumerate(VOCAB)}

def load_dataset(lab_csv="./dataset/labeled/labels.csv"):
    df = pd.read_csv(lab_csv)
    X, Y = [], [[] for _ in range(T)]
    kept, drop_len, drop_vocab, miss = 0,0,0,0
    for _, row in df.iterrows():
        path = pathlib.Path("./dataset/raw")/row["filename"]
        if not path.exists():
            # fallback: labeled kopyadan al
            path = pathlib.Path("./dataset/labeled")/(path.stem+"__"+row["label"]+".png")
        if not path.exists():
            miss += 1; continue
        try:
            img = Image.open(path).convert("RGB")
        except:
            miss += 1; continue
        img = ImageOps.contain(img, (W,H), method=Image.LANCZOS)
        canvas = Image.new("RGB", (W,H), (0,0,0))
        canvas.paste(img, ((W-img.width)//2, (H-img.height)//2))
        a = np.asarray(canvas, dtype=np.float32)/255.0
        label = str(row["label"]).strip()
        if len(label)!=T:
            drop_len+=1; continue
        try:
            for ch in label:
                if ch not in MAP: raise ValueError
        except:
            drop_vocab+=1; continue
        X.append(a)
        for i,ch in enumerate(label):
            Y[i].append(MAP[ch])
        kept+=1
    X = np.stack(X, axis=0)
    Y = [np.array(y,dtype=np.int32) for y in Y]
    print(f"[DATA] kept={kept}  dropped_len={drop_len}  dropped_vocab={drop_vocab}  missing_files={miss}")
    return X, Y

def build_model_with_heads(backbone, vocab_size=len(VOCAB), T=T):
    x = backbone.output
    if isinstance(x, list):  # bazı modeller list dönebilir
        x = x[0]
    if len(x.shape)==4:
        x = layers.GlobalAveragePooling2D()(x)
    outs = [layers.Dense(vocab_size, activation="softmax", name=f"c{i}")(x) for i in range(T)]
    return models.Model(backbone.input, outs)

def compile_for_training(model, lr=1e-3):
    losses = {f"c{i}":"sparse_categorical_crossentropy" for i in range(T)}
    metrics = {f"c{i}":"accuracy" for i in range(T)}
    model.compile(optimizer=keras.optimizers.Adam(lr), loss=losses, metrics=metrics)

def train_head_only(X,Y, backbone_path, out_path):
    base = keras.models.load_model(backbone_path, compile=False)
    for l in base.layers: l.trainable=False
    model = build_model_with_heads(base)
    compile_for_training(model, lr=1e-3)
    model.fit(
        X, {f"c{i}":Y[i] for i in range(T)},
        batch_size=64, epochs=10, validation_split=0.1,
        callbacks=[keras.callbacks.ModelCheckpoint(out_path, save_best_only=True)]
    )
    return model

def finetune_backbone(model, X,Y, out_path):
    # üst katmanların bir kısmını aç
    for l in model.layers[-8:]:
        l.trainable=True
    compile_for_training(model, lr=5e-5)
    model.fit(
        X, {f"c{i}":Y[i] for i in range(T)},
        batch_size=64, epochs=5, validation_split=0.1,
        callbacks=[keras.callbacks.ModelCheckpoint(out_path, save_best_only=True)]
    )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="./my_model.keras", help="önceden eğitilmiş model yolu")
    ap.add_argument("--out", default="./ft_model.keras", help="çıktı model yolu")
    ap.add_argument("--labels", default="./dataset/labeled/labels.csv")
    args = ap.parse_args()

    X,Y = load_dataset(args.labels)
    model = train_head_only(X,Y, args.backbone, args.out)
    finetune_backbone(model, X,Y, args.out)
    print(f"[DONE] Best model saved to: {args.out}")

if __name__=="__main__":
    main()
