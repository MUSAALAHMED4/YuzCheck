import tkinter as tk
import subprocess
from PIL import Image, ImageTk

# Uygulama Arayüzü
root = tk.Tk()
root.title("YuzCheck - Dosya Yöneticisi")
root.geometry("500x500")
root.configure(bg="#2C2F33")  # Arka plan rengi
root.minsize(400, 500)  # Minimum pencere boyutu

# Düğme stilleri
button_style = {
    "font": ("Arial", 14, "bold"),
    "bg": "#7289DA",
    "fg": "white",
    "width": 35,
    "height": 2,
    "bd": 3,
    "relief": "ridge"
}

def run_pdf_extractor():
    subprocess.Popen(["python", "pdfFotocikar.py"])

def run_image_renamer():
    subprocess.Popen(["python", "Resimlerinadlarını degistirme.py"])

# Ana çerçeve
main_frame = tk.Frame(root, bg="#2C2F33")
main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

# Logo yükleme
logo_path = "logo.png"  # Logonun bulunduğundan emin olun
try:
    logo_img = Image.open(logo_path)
    logo_img = logo_img.resize((100, 100))
    logo_tk = ImageTk.PhotoImage(logo_img)
    logo_label = tk.Label(main_frame, image=logo_tk, bg="#2C2F33")
    logo_label.pack(pady=10)
except Exception as e:
    print("Logo yükleme hatası:", e)

# Başlık etiketi
title_label = tk.Label(main_frame, text="YuzCheck - Dosya Yöneticisi", font=("Arial", 18, "bold"), bg="#2C2F33", fg="white")
title_label.pack(pady=5)

# Düğme çerçevesi
buttons_frame = tk.Frame(main_frame, bg="#2C2F33")
buttons_frame.pack(fill=tk.BOTH, expand=True)

# Düğmeler
download_pdf_btn = tk.Button(buttons_frame, text="📂 PDF'ten Fotoğraf Çıkar", command=run_pdf_extractor, **button_style)
download_pdf_btn.pack(pady=10, fill=tk.X)

rename_images_btn = tk.Button(buttons_frame, text="📷 Fotoğraf İsimlerini Güncelle", command=run_image_renamer, **button_style)
rename_images_btn.pack(pady=10, fill=tk.X)

# Takım üyeleri
team_label = tk.Label(root, text="Geliştirenler:\nMusa Alahmed 23040301118\nYazen Emino 22040301111\nMuhammed Jalahej 22040301083", font=("Arial", 10), bg="#2C2F33", fg="white", justify="center")
team_label.pack(pady=10)

# Ana pencereyi çalıştır
root.mainloop()
