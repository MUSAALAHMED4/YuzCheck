import os
import pandas as pd
import shutil

# 📌 Dosya yollarını belirleme
extract_dir = "cikarilan_fotograflar"  
new_photos_dir = "updated_photos" 
excel_path = "ogrenci_listesi_birlesik.xlsx"  

# 📌 Fotoğraf klasörünün var olup olmadığını kontrol et
if not os.path.exists(extract_dir):
    print(f"⚠️ Klasör bulunamadı: {extract_dir}. Lütfen fotoğrafları bu klasöre koyun.")
    exit()

# 📌 Excel dosyasını oku
excel_data = pd.read_excel(excel_path)

# 📌 Kolon isimlerinden gereksiz boşlukları temizle
excel_data.columns = excel_data.columns.str.strip()

# 📌 Excel'deki mevcut sütun isimlerini kontrol et
print("📌 Excel dosyasındaki sütunlar:")
print(excel_data.columns.tolist())

# 📌 Doğru sütun isimlerini belirle
number_col = "Öğrenci Numarası"  # Öğrenci numarası sütunu
name_col = "Öğrenci Adı"  # Öğrenci adı sütunu

# 📌 Öğrenci numaralarını temizle ve metin formatına çevir
excel_data[number_col] = excel_data[number_col].astype(str).str.strip()

# 📌 {Öğrenci numarası: Öğrenci adı} sözlüğünü oluştur
student_dict = {str(k).strip(): v.strip().replace(" ", "_") for k, v in zip(excel_data[number_col], excel_data[name_col])}

# 📌 Kontrol için bazı değerleri yazdır
print("📌 Excel'deki öğrenci numaraları (eşleşmeyi kontrol etmek için):")
print(list(student_dict.keys())[:10])  # İlk 10 değeri yazdır

# 📌 Yeni fotoğraf klasörünü oluştur (eğer yoksa)
os.makedirs(new_photos_dir, exist_ok=True)

# 📌 Fotoğrafları yeniden adlandır ve yeni klasöre taşı
for filename in os.listdir(extract_dir):
    file_path = os.path.join(extract_dir, filename)
    if os.path.isfile(file_path):
        # Dosya adından öğrenci numarasını çıkar
        student_number = "".join(filter(str.isdigit, filename)).strip()  # Sadece rakamları al
        
        print(f"  Fotoğraf inceleniyor: {filename} | Çıkarılan öğrenci numarası: {student_number}")
        
        if student_number in student_dict:
            correct_name = student_dict[student_number]  # Doğru ismi al
            new_filename = f"{correct_name}_{student_number}.jpg"
            new_file_path = os.path.join(new_photos_dir, new_filename)
            shutil.copy2(file_path, new_file_path)  # Yeni isimle kopyala
            print(f"✅ Yeniden adlandırıldı: {filename} → {new_filename}")
        else:
            print(f"⚠️ Öğrenci numarası bulunamadı: {student_number}, orijinal isim korunuyor.")
            new_file_path = os.path.join(new_photos_dir, filename)
            shutil.copy2(file_path, new_file_path)  # Eşleşme yoksa orijinal ismi koru

print(f"🎉 Fotoğraflar başarıyla güncellendi! Yeni adlandırılan fotoğrafları şu klasörde bulabilirsiniz: {new_photos_dir}")