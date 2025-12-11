import tkinter as tk
import subprocess
from PIL import Image, ImageTk

# Uygulama Arayüzünü Geliştirme
root = tk.Tk()
root.title("YuzCheck - Yüz Tanıma Projesi")
root.geometry("500x500")
root.configure(bg="#2C2F33")  # Arka plan rengini değiştir
root.minsize(400, 500)  # İçeriğin kaybolmamasını sağlamak için minimum pencere boyutu

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

def run_interface1():
    subprocess.Popen(["python", "readimages-fordatabase.py"])

def run_interface2():
    subprocess.Popen(["python", "winwebcam.py"])

def run_interface3():
    subprocess.Popen(["python", "vidileyoklama.py"])


# Pencere boyutu değiştiğinde düzeni korumak için ana çerçeve
main_frame = tk.Frame(root, bg="#2C2F33")
main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

# Logo yükleme
logo_path = "logo.png"  # Resmin proje klasöründe olduğundan emin olun
try:
    logo_img = Image.open(logo_path)
    logo_img = logo_img.resize((100, 100))  # Image.ANTIALIAS kaldırıldı çünkü yeni sürümlerde desteklenmiyor
    logo_tk = ImageTk.PhotoImage(logo_img)
    logo_label = tk.Label(main_frame, image=logo_tk, bg="#2C2F33")
    logo_label.pack(pady=10)
except Exception as e:
    print("Logo yükleme hatası:", e)

# Başlık etiketi
title_label = tk.Label(main_frame, text="YuzCheck - Yüz Tanıma Projesi", font=("Arial", 18, "bold"), bg="#2C2F33", fg="white")
title_label.pack(pady=5)

# Düğmelerin düzenli ve uyumlu olması için iç çerçeve
buttons_frame = tk.Frame(main_frame, bg="#2C2F33")
buttons_frame.pack(fill=tk.BOTH, expand=True)

# Düğmelerin eşit şekilde yerleşmesini sağlama
buttons_frame.grid_rowconfigure(0, weight=1)
buttons_frame.grid_rowconfigure(1, weight=1)
buttons_frame.grid_rowconfigure(2, weight=1)
buttons_frame.grid_rowconfigure(3, weight=1)
buttons_frame.grid_columnconfigure(0, weight=1)

# Arayüzler arasında geçiş yapmak için düğmeler
btn1 = tk.Button(buttons_frame, text="Yüz Tespiti", command=run_interface1, **button_style)
btn1.grid(row=0, column=0, pady=10, sticky="ew")

btn2 = tk.Button(buttons_frame, text="Görüntü İşleme", command=run_interface2, **button_style)
btn2.grid(row=1, column=0, pady=10, sticky="ew")

btn3 = tk.Button(buttons_frame, text="Yoklama Takibi", command=run_interface3, **button_style)
btn3.grid(row=2, column=0, pady=10, sticky="ew")


# Takım üyelerini ekleme
team_label = tk.Label(root, text="Geliştirenler:\nMusa Alahmed 23040301118\nYazen Emino 22040301111\nMuhammed Jalahej 22040301083", font=("Arial", 10), bg="#2C2F33", fg="white", justify="center")
team_label.pack(pady=10)

# Ana pencereyi çalıştır
root.mainloop()