# train_scratch.py
# RGB 50x200, sabit uzunluk=5, VOCAB=0-9 A-Z a-z
# Keras / TF 2.x

import os, pathlib, numpy as np, pandas as pd
import keras
from keras import layers, models, callbacks, optimizers
from PIL import Image, ImageOps

# ==== Ayarlar ====
H, W = 50, 200
T = 5
VOCAB = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
MAP = {ch:i for i,ch in enumerate(VOCAB)}
NUM_CLASSES = len(VOCAB)

LABELS_CSV = "./dataset/labeled/labels.csv"
RAW_DIR    = pathlib.Path("./dataset/raw")
OUT_PATH   = "./captcha_scratch.keras"

BATCH_SIZE = 64
EPOCHS     = 40
VAL_SPLIT  = 0.12
SEED       = 42

# Augment düzeyi (hafif orta)
AUG = keras.Sequential([
    layers.RandomContrast(0.2),
    layers.RandomBrightness(factor=0.2),
    layers.RandomTranslation(0.03, 0.05, fill_mode="nearest"),
    layers.RandomZoom((-0.05, 0.12), fill_mode="nearest"),
], name="augment")
# ===============

def letterbox_rgb(img_pil: Image.Image, size=(W, H)) -> np.ndarray:
    rgb = img_pil.convert("RGB")
    rgb = ImageOps.contain(rgb, size, method=Image.LANCZOS)
    canvas = Image.new("RGB", size, (0,0,0))
    canvas.paste(rgb, ((size[0]-rgb.width)//2, (size[1]-rgb.height)//2))
    a = np.asarray(canvas, dtype=np.float32) / 255.0
    return a

def load_dataset():
    df = pd.read_csv(LABELS_CSV, dtype={"filename":str, "label":str})
    X, Y = [], [[] for _ in range(T)]
    kept=dropped_len=dropped_vocab=missing=0
    for _, r in df.iterrows():
        fn  = (r["filename"] or "").strip()
        lab = (r["label"] or "").strip()
        p = RAW_DIR/fn
        if not p.exists():
            missing += 1; continue
        if len(lab)!=T:
            dropped_len += 1; continue
        if any(ch not in MAP for ch in lab):
            dropped_vocab += 1; continue
        try:
            img = Image.open(p)
            X.append(letterbox_rgb(img, (W,H)))
            for i,ch in enumerate(lab):
                Y[i].append(MAP[ch])
            kept += 1
        except Exception:
            pass
    if kept==0: raise RuntimeError("No valid samples.")
    X = np.stack(X,0).astype(np.float32)              # (N,H,W,3)
    Y = [np.array(y, np.int32) for y in Y]            # 5 × (N,)
    print(f"[DATA] kept={kept} drop_len={dropped_len} drop_vocab={dropped_vocab} miss={missing}")
    return X, Y

def conv_block(x, f, k=3, s=1, p="same", dw=False, act=True):
    if dw:
        x = layers.SeparableConv2D(f, k, strides=s, padding=p, use_bias=False)(x)
    else:
        x = layers.Conv2D(f, k, strides=s, padding=p, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    if act: x = layers.Activation("relu")(x)
    return x

def build_model():
    inp = layers.Input((H,W,3))
    x = conv_block(inp, 32, 3);        x = conv_block(x, 32, 3); x = layers.MaxPool2D((2,2))(x)   # 25x100
    x = conv_block(x, 64, 3, dw=True); x = conv_block(x, 64, 3); x = layers.MaxPool2D((2,2))(x)   # 12x50
    x = conv_block(x, 96, 3, dw=True); x = conv_block(x, 96, 3); x = layers.MaxPool2D((2,2))(x)   # 6x25
    x = conv_block(x, 128, 3, dw=True);x = conv_block(x, 128, 3)
    x = layers.Dropout(0.2)(x)
    x = layers.GlobalAveragePooling2D(name="gap")(x)   # (B,128)

    outs = [layers.Dense(NUM_CLASSES, activation="softmax", name=f"c{i}")(x) for i in range(T)]
    return models.Model(inp, outs)

def compile_model(m, lr=3e-4):
    losses  = {f"c{i}": keras.losses.SparseCategoricalCrossentropy() for i in range(T)}
    metrics = {f"c{i}": "accuracy" for i in range(T)}
    opt = optimizers.Adam(learning_rate=lr)
    m.compile(optimizer=opt, loss=losses, metrics=metrics)

def split_train_val(X, Y, val_split=VAL_SPLIT):
    np.random.seed(SEED)
    idx = np.random.permutation(len(X))
    X = X[idx]; Y = [y[idx] for y in Y]
    n = len(X); n_val = max(1, int(n*val_split))
    Xv, Xt = X[:n_val], X[n_val:]
    Yv = [y[:n_val] for y in Y]; Yt = [y[n_val:] for y in Y]
    return (Xt,Yt), (Xv,Yv)

def gen_aug(Xa, Ya, batch=BATCH_SIZE):
    n=len(Xa); i=0
    while True:
        j=min(i+batch,n)
        xb = Xa[i:j]
        # augment sadece train’de
        xb = AUG(xb, training=True).numpy()
        yb = {f"c{k}": Ya[k][i:j] for k in range(T)}
        yield xb, yb
        i = 0 if j>=n else j

def exact_match_metric(y_true_list, y_pred_list):
    # y_true_list: [N], ... 5 tane; y_pred_list: logits listesi
    pred_ids = [np.argmax(p, axis=1) for p in y_pred_list]  # 5 × (N,)
    N = pred_ids[0].shape[0]
    em = 0
    for n in range(N):
        ok = True
        for t in range(T):
            if pred_ids[t][n] != y_true_list[t][n]:
                ok = False; break
        if ok: em += 1
    return em / N

def main():
    X, Y = load_dataset()
    (Xt,Yt), (Xv,Yv) = split_train_val(X, Y)

    model = build_model()
    compile_model(model, lr=3e-4)

    cbs = [
        callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-5, verbose=1),
        callbacks.ModelCheckpoint(OUT_PATH, monitor="val_loss", save_best_only=True, verbose=1),
    ]

    steps = int(np.ceil(len(Xt)/BATCH_SIZE))
    val = (Xv, {f"c{i}": Yv[i] for i in range(T)})
    model.fit(gen_aug(Xt,Yt), steps_per_epoch=steps, validation_data=val, epochs=EPOCHS, callbacks=cbs, verbose=1)

    # Val exact-match raporu
    preds = model.predict(Xv, verbose=0)
    em = exact_match_metric(Yv, preds)
    print(f"[VAL] exact-match@5 = {em:.4f}")

    try: model.save(OUT_PATH)
    except Exception: pass
    print(f"[DONE] saved -> {OUT_PATH}")

if __name__ == "__main__":
    main()
