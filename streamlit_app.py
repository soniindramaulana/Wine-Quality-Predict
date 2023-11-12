import numpy as np
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import joblib
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
import streamlit as st
from sklearn.ensemble import AdaBoostClassifier


st.set_page_config(
    page_title="Wine Quality Prediction",
    page_icon="🍷",
    layout="wide"
)

st.title("🍷 Wine Quality Prediction")
st.write('''
*Penjelasan*

Berikut merupakan data set hasil uji oleh badan sertifikasi resmi (CVRVV) yang telah mengumpulkan data dari tanggal Mei 2004 hingga Februari 2007.
Sampel uji berupa data kandungan anggur dari perusahaan Vinho Verde di Minho (barat laut) Portugal. Terdapat 2 jenis varian yang telah dikumpulkan datanya yaitu varian merah dan putih.
pada penelitian kali ini, akan digunakan data varian merah dengan jumlah data sebanyak 1559. yang bersikan 1 kolom sebagai label kualitas dari anggur tersebut dimulai dari 0 (sangat buruk) hingga 10 (sangat baik).
Tentu saja kualitas dari anggur didukung oleh keberadaan 11 indikator/fitur nya yang seluruhnya beritipe rasio decimal.
''')
# Membaca data dari file CSV
data = pd.read_csv('winequality-red.csv')
st.write(data)

st.write('''
---
Daftar Fitur :


1. *Fixed Acidity (4 - 6 (g/dm3))* : Merupakan kadar asam tartarat pada anggur. Asam tetap yang dominan ditemukan dalam anggur adalah tartarat, malat, sitrat, dan suksinat.Tingkat masing-masing yang ditemukan dalam anggur bisa sangat bervariasi tetapi dalam secara umum orang akan mengharapkan untuk melihat 1.000 hingga 4.000 mg/L asam tartarat, 0 hingga 8.000 mg/L asam malat, 0 hingga 500 mg/L asam sitrat, dan 500 hingga 2.000 mg/L asam suksinat.
2. *Volatile Acidity (0,5 - 0,7 (g/dm³))* : Kadar asam asetat pada anggur.  Keasaman yang mudah menguap adalah ukuran berat molekul rendah (atau uap dapat disuling) asam lemak dalam anggur dan umumnya dianggap sebagai bau cuka. Rata-rata nilai keasaman yang mudah menguap untuk anggur meja merah selama periode ini adalah sekitar 0,60 gram/L.
3. *Citric Acid (0,5 - 0,75 (g/dm³))* : Kadar asam sitrat dalam anggur.  Asam sitrat memiliki banyak kegunaan dalam produksi anggur. Asam sitrat adalah bahan organik lemah asam, yang sering digunakan sebagai pengawet alami atau bahan tambahan pada makanan atau minuman untuk menambah rasa asam pada makanan
4. *Residual Sugar (4 - 20 (g/dm3))* : Kadar sisa gula pada anggur. Gula Residu berasal dari gula anggur alami yang tersisa dalam anggur setelahnya fermentasi alkohol selesai. Itu diukur dalam gram per liter. Sisa kadar gula bervariasi pada berbagai jenis anggur. Faktanya, banyak toko kelontong anggur berlabel "kering" mengandung sekitar 10 g/L sisa gula.
5. *Chlorides  (0,20 - 0,60 (g/dm3))* : Kadar natrium klorida pada anggur. Klorida (natrium klorida) memberi rasa asin pada anggur yang bisa berubah menjauhkan konsumen potensial. Natrium klorida, umumnya dikenal sebagai garam, adalah senyawa ionik dengan rumus kimia NaCl. Maksimal konsentrasi klorida dalam anggur sekitar 0,20 - 0,60 g/L (Vallone et al., 2021).
6. *Free sulfur dioxide  (30 - 50 (mg/dm3))* : Tingkat sulfur dioksida bebas pada anggur. Sulfur dioksida (SO2) dan garamnya telah ditambahkan selama pembuatan anggur sejak abad ke-17. Ini masih berpotensi menyebabkan reaksi merugikan dan produk beracun bagi konsumen anggur dan pembuat anggur dalam jumlah lebih dari 10 mg/L, dan karenanya, seharusnya ditangani dengan hati-hati.
7. *Total sulfur dioxide  (kurang dari 100 (mg/dm3))* : merupakan SO2 Total yang dihasilkan dari penjumlahan SO2 bebas dan SO2 bereaksi. Bagian dari sulfur dioksida bebas ditambah bagian yang terikat dengan bahan kimia lain di dalamnya anggur seperti aldehida, pigmen, atau gula konsentrasi nilai SO2 bebas 25 mg/L aktif anggur merah. SO2 yang aktif konsentrasi 0,35 mg/L memastikan perlindungan minimum, dan nilai Perlindungan maksimal 0,6 mg/L
8. *Density  (1,00 - 1,05 (g/cm3))* : merupakan kepadatan massa jenis anggur. Massa jenis adalah massa per satuan volume anggur atau harus pada 20◦C. Dia dinyatakan dalam gram per mililiter, dan dilambangkan dengan simbol 20◦C.
9. *pH  (3-4)* : merupakan tingkat keasaman anggur. pH adalah skala yang digunakan untuk menentukan keasaman atau kebasaan suatu larutan berair (pH yang lebih rendah menunjukkan keasaman yang lebih tinggi). Tingkat pH anggur berkisar antara 3 hingga 4
10. *Sulphates  (0,1 - 0,2 (g/dm3))* : merupakan kadar kalium sulfat dalam anggur. Sulfit, juga biasa disebut sulfur dioksida, adalah senyawa kimia yang mengandung ion sulfit. Mereka ditemukan secara alami dalam berbagai makanan sumbernya, termasuk teh hitam, kacang tanah, telur, dan makanan fermentasi. Meskipun mereka ditemukan di banyak makanan dan minuman, hal ini terutama terkait dengan daftar panjang efek samping yang berkaitan dengan anggur konsumsi, termasuk sakit kepala akibat anggur yang ditakuti (Roullier-Gall dkk., 2017).
11. *Alcohol  (5 - 23 (% vol))* : merupakan konsentrasi alkohol dalam anggur. Alkohol adalah senyawa organik yang membawa setidaknya satu hidroksil gugus fungsi (−OH) terikat pada atom karbon jenuh. Anggur bisa dimiliki berkisar antara 5% dan 23% Alkohol berdasarkan Volume (ABV). Output variable (based on sensory data):

## Tujuan
Tujuan dari penelitian ini yaitu melakukan prediksi tehadap kualitas dari anggur merah Vinho Verde yang didukung oleh keberadaan fitur/indikator yang mempengaruhi dari kualitas anggur merah.
Data uji prediksi nantinya akan melebeli data menjadi salah satu dari 2 kelas yang berbeda
yaitu kelas 1 untuk data anggur dengan kualitas yang baik lalu kelas 0 untuk data dengan kelas buruk.''')

dt = pd.read_csv('data_clean.csv')
fitur_balance = dt.drop(columns=['quality'], axis= 1)
target_balance = dt['quality']

st.title("Prediksi")
Fixed_Acidity = st.number_input("Kandungan Asam Tartarat (4 - 6 g/dm3): ")
Volatile_Acidity = st.number_input("Kandungan Asam Asetat (0,5 - 0,7 (g/dm³)) : ")
Citric_Acid = st.number_input("Kandungan Asam Sitrat (0,5 - 0,75 (g/dm³)): ")
Residual_Sugar = st.number_input("Kandungan Gula Sisa (4 - 20 (g/dm3)) : ")
Chlorides = st.number_input("Kandungan Natrium Klorida (0,20 - 0,60 (g/dm3)) : ")
Free_Sulfur_Dioxide = st.number_input("Kandungan SO2 Bebas   (30 - 50 (mg/dm3)) : ")
Total_Sulfur_Dioxide = st.number_input("Kandungan SO2 Total   (kurang dari 100 (mg/dm3)) : ")
Density = st.number_input("Konsentrasi Massa Jenis (1,00 - 1,05 (g/cm3)) : ")
pH = st.number_input("Tingkat Keasaman (3-4) : ")
Sulphates = st.number_input("Kandungan Kalium Sulfat (0,1 - 0,2 (g/dm3)) : ")
Alcohol = st.number_input("Volume Alcohol (5 - 23 (% vol)) : ")

results = []
data = {'fixed acidity' : Fixed_Acidity,
        'volatile acidity' : Volatile_Acidity,
        'citric acid' : Citric_Acid,
        'residual sugar' : Residual_Sugar,
        'chlorides' : Chlorides,
        'free sulfur dioxide' : Free_Sulfur_Dioxide,
        'total sulfur dioxide' : Total_Sulfur_Dioxide,
        'density' : Density,
        'pH' : pH,
        'sulphates' : Sulphates,
        'alcohol' : Alcohol
        }

results.append(data)
data_implementasi = pd.DataFrame(results)

if st.button("Cek Prediksi"):
    fitur_train, fitur_test, target_train, target_test = train_test_split(fitur_balance, target_balance, test_size=0.25, random_state=42)

    st.write(data_implementasi)
    minmaxscaler = joblib.load('minmax.pkl')

    minmaxscaler.fit(fitur_train)
    minmaxtraining = minmaxscaler.transform(fitur_train)
    minmaxtesting = minmaxscaler.transform(fitur_test)

    rfmodel = joblib.load('model_rf_minmax.pkl')
    rfmodel.fit(minmaxtraining, target_train)

    prediksi = rfmodel.predict(data_implementasi)
    st.text('Prediksi : ')
    prediksi
    if prediksi[0] == 0:
        st.warning("Kualitas Wine Anda termasuk dalam kategori Buruk !")
    elif prediksi[0] == 1:
        st.success("Kualitas Wine Anda termasuk dalam kategori Baik  !")
