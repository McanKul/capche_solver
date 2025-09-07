# labeler.py
import csv, shutil, pathlib, sys
from tkinter import Tk, Label, Entry, Button, StringVar
from PIL import Image, ImageTk

RAW_DIR = pathlib.Path("./dataset/raw")
LAB_DIR = pathlib.Path("./dataset/labeled"); LAB_DIR.mkdir(parents=True, exist_ok=True)
CSV_PATH = LAB_DIR / "labels.csv"

def main():
    files = sorted(RAW_DIR.glob("*.png"))
    if not files:
        print("No images found in ./dataset/raw")
        sys.exit(1)

    # if CSV exists, skip already-labeled
    labeled = set()
    if CSV_PATH.exists():
        with open(CSV_PATH, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                labeled.add(row["filename"])
        files = [f for f in files if f.name not in labeled]

    idx = 0

    root = Tk()
    root.title("CAPTCHA Labeler (minimal)")

    img_label = Label(root)
    img_label.pack(padx=10, pady=10)

    text_var = StringVar()
    entry = Entry(root, textvariable=text_var, font=("Consolas", 18), width=16)
    entry.pack(padx=10, pady=6)

    status = Label(root, text="")
    status.pack(pady=4)

    def load_image(i):
        if i >= len(files):
            img_label.config(image="", text="Done!")
            status.config(text=f"Labeled all. CSV: {CSV_PATH}")
            entry.config(state="disabled")
            return
        im = Image.open(files[i])
        imgtk = ImageTk.PhotoImage(im)
        img_label.imgtk = imgtk
        img_label.config(image=imgtk)
        status.config(text=f"{i+1}/{len(files)}  |  {files[i].name}")
        entry.delete(0, "end")
        entry.focus_set()

    def append_csv(filename, label):
        write_header = not CSV_PATH.exists()
        with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if write_header:
                w.writerow(["filename", "label"])
            w.writerow([filename, label])

    def save_and_next(event=None):
        nonlocal idx
        if idx >= len(files): return
        label = text_var.get().strip()
        if not label: return

        # write to CSV
        append_csv(files[idx].name, label)

        # copy labeled image for quick visual checking
        dst = LAB_DIR / f"{files[idx].stem}__{label}.png"
        shutil.copy2(files[idx], dst)

        idx += 1
        load_image(idx)

    entry.bind("<Return>", save_and_next)
    Button(root, text="Save (Enter)", command=save_and_next).pack(pady=6)

    load_image(idx)
    root.mainloop()

if __name__ == "__main__":
    main()
