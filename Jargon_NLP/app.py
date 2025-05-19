from flask import Flask, render_template, request
import pandas as pd
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
from deep_translator import GoogleTranslator

app = Flask(__name__)

# Model ve tokenizer yüklenme kısmı
model = tf.keras.models.load_model("text_model.h5")
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# Yazılım terimleri ve negatif kelimeler listesinin yüklenmesi aşaması
terim_df = pd.read_csv("yazılım_kelimeler.csv")
terim_df['kelime'] = terim_df['kelime'].str.strip().str.lower()
terim_kumesi = set(terim_df['kelime'])

negatif_df = pd.read_csv("negatif_kelimelerson.csv")
negatif_df['kelime'] = negatif_df['kelime'].str.strip().str.lower()
negatif_kumesi = set(negatif_df['kelime'])

# Temizleme fonksiyonu
def temizle(kelime):
    kelime = kelime.strip().lower()
    kelime = re.sub(r"[^\w.]", "", kelime)
    return kelime

@app.route("/", methods=["GET", "POST"])
def index():
    ortalama_dogruluk = 0
    bulunanlar = []
    negatif_bulunanlar = []
    metin = ""

    if request.method == "POST":
        metin = request.form["metin"].lower()
        ceviri_istegi = "translate" in request.form
        kelimeler = metin.split()

        islenen_kelimeler = set()
        dogruluk_toplam = 0
        dogru_sayisi = 0

        for kelime in kelimeler:
            kelime_temiz = temizle(kelime)

            if kelime_temiz == "" or kelime_temiz in islenen_kelimeler:
                continue
            islenen_kelimeler.add(kelime_temiz)

            # Negatif kelime kontrolü: eğer kelime negatif listede ise yapılacak işlemler
            if kelime_temiz in negatif_kumesi:
                negatif_bulunanlar.append(kelime_temiz)
                continue 

            # Model tahmininin yapıldığı kısım
            seq = tokenizer.texts_to_sequences([kelime_temiz])
            if not seq or not seq[0]:
                continue

            seq = pad_sequences(seq, maxlen=model.input_shape[1])
            tahmin = model.predict(seq, verbose=0)[0][0]

            # Sadece yazılım terimi olarak tahmin edilen kelimelerin eklendiği kısım
            if tahmin >= 0.5:
                dogru_sayisi += 1
                dogruluk_toplam += tahmin

                # CSV dosyamda kelimenin açıklaması var mı yok mu kontrolü
                if kelime_temiz in terim_kumesi:
                    aciklama = terim_df[terim_df['kelime'] == kelime_temiz]['aciklama'].values[0]
                else:
                    aciklama = "Bu yazılım terimi açıklama listesinde yok."

                # isteğe bağlı olarak çeviri yapılması kısmı
                if ceviri_istegi:
                    try:
                        ceviri = GoogleTranslator(source='auto', target='tr').translate(kelime_temiz)
                        aciklama += f" (Çeviri: {ceviri})"
                    except Exception:
                        aciklama += " (Çeviri hatası)"

                bulunanlar.append((kelime_temiz, aciklama))

            else:
                # Model girilen kelimeyi yazılım terimi olarak tahmin etmediyse ve kelime negatif_kelimeler csv dosyasında da değilse,
                # o zaman model bunu negatif olarak değil sadece görmezden geliyor.çıktı olarak vermiyor.
                # bu aşama proje geliştirildikçe değiştirilecek.
                pass

        if dogru_sayisi > 0:
            ortalama_dogruluk = dogruluk_toplam / dogru_sayisi

    return render_template(
        "index.html",
        bulunanlar=bulunanlar,
        negatif_bulunanlar=negatif_bulunanlar,
        metin=metin,
        ortalama_dogruluk=ortalama_dogruluk
    )

if __name__ == "__main__":
    app.run(debug=False)
