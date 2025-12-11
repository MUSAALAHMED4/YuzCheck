import fitz  # PyMuPDF
import os
import cv2
import numpy as np
import re

# 📌 PDF dosyasını belirleme
pdf_dosyasi = "2.pdf" 

# 📌 Çıkarılan resimleri kaydetmek için klasör oluşturma
cikis_klasoru = "cikarilan_fotograflar"
if not os.path.exists(cikis_klasoru):
    os.makedirs(cikis_klasoru)

# 📌 PDF dosyasını açma
pdf_belgesi = fitz.open(pdf_dosyasi)

# 📌 PDF'den metinleri çıkarma
tum_metin = ""
for sayfa in pdf_belgesi:
    tum_metin += sayfa.get_text("text", sort=True).encode('utf-8', 'ignore').decode('utf-8') + "\n"

# 📌 Metinlerden sadece öğrenci numaralarını çıkarmak için regex kullanma
pattern = r"(\d{11})"
ogrenci_numaralari = re.findall(pattern, tum_metin)

# 📌 Resimleri çıkarma ve öğrenci numaraları ile eşleştirme
indeks = 0
for sayfa_numarasi in range(len(pdf_belgesi)):
    sayfa = pdf_belgesi[sayfa_numarasi]
    resimler = sayfa.get_images(full=True)

    for resim_indeks, resim in enumerate(resimler):
        xref = resim[0]
        temel_resim = pdf_belgesi.extract_image(xref)
        resim_verisi = temel_resim["image"]
        uzanti = temel_resim["ext"]
        genislik = temel_resim["width"]
        yukseklik = temel_resim["height"]

        # 🔹 *Küçük resimleri  yok sayma*
        if genislik < 150 or yukseklik < 150:
            print(f"⏩ Küçük resim yok sayıldı (muhtemelen logo): {genislik}x{yukseklik}")
            continue

        # 🔹 *Resimleri öğrenci numaralarıyla eşleştirme*
        if indeks < len(ogrenci_numaralari):
            ogrenci_num = ogrenci_numaralari[indeks]
            indeks += 1
        else:
            ogrenci_num = f"Bilinmeyen_{sayfa_numarasi + 1}_{resim_indeks + 1}"

        # 📌 Resmi numpy array'e dönüştürme
        resim_dizisi = np.frombuffer(resim_verisi, dtype=np.uint8)
        resim_nesnesi = cv2.imdecode(resim_dizisi, cv2.IMREAD_COLOR)

        if resim_nesnesi is None:
            print(f"⏩ Hata: Resim yüklenemedi -> {ogrenci_num}")
            continue  # Bozuk resimleri atla

        # 📌 Resmi sadece öğrenci numarası ile kaydetme
        resim_adi = f"{ogrenci_num}.{uzanti}"
        resim_yolu = os.path.join(cikis_klasoru, resim_adi)
        cv2.imwrite(resim_yolu, resim_nesnesi)

        print(f"✅ Resim kaydedildi: {resim_yolu}")

print("🎉 Resimler çıkarıldı ve yalnızca öğrenci numaraları ile kaydedildi!")
