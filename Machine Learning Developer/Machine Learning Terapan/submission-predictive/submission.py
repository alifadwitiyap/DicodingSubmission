# %% [markdown]
# ## Nama : Alif Adwitiya Pratama

# %% [markdown]
# # Predictive Analysis - Diabetes Risk Prediction

# %% [markdown]
# ### 1. Persiapan

# %% [markdown]
# #### 1.1 Masukkan Library

# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV


#remove warning
import warnings
warnings.filterwarnings('ignore')

# %% [markdown]
# #### 1.2 Masukkan Data

# %%
df=pd.read_csv('./data/diabetes.csv')

# %% [markdown]
# ### 2. Data Understanding

# %% [markdown]
# dataset yang digunakan dapat diakses [disini](https://www.kaggle.com/datasets/jillanisofttech/diabetes-disease-updated-dataset)

# %% [markdown]
# #### 2.1 Tentang Dataset

# %% [markdown]
# **Deskripsi**<br>
# Dataset ini berasal dari National Institute of Diabetes and Digestive and Kidney Diseases. Tujuannya adalah untuk memprediksi berdasarkan pengukuran diagnostik apakah seorang pasien menderita diabetes. Batasan dataset ini adalah bahwa semua pasien adalah wanita berusia 21 tahun atau lebih dengan keturunan Pima Indian.
# diambil dari [sini](https://www.kaggle.com/datasets/jillanisofttech/diabetes-disease-updated-dataset)
# 
# **Tentang Fitur** <br>
# terdapat 8 fitur yang digunakan untuk memprediksi apakah seseorang menderita diabetes atau tidak. Fitur-fitur tersebut adalah:
# - Pregnancies: Jumlah kehamilan
# - Glucose: Konsentrasi glukosa plasma 2 jam dalam tes toleransi glukosa oral
# - BloodPressure: Tekanan darah diastolik (mm Hg)
# - SkinThickness: Ketebalan lipatan kulit trisep (mm)
# - Insulin: Insulin serum 2 jam (mu U / ml)
# - BMI: Indeks massa tubuh (berat dalam kg / (tinggi dalam m) ^ 2)
# - DiabetesPedigreeFunction: Fungsi silsilah diabetes
# - Age: Usia (tahun)
# 
# **Tentang Target**<br>
# Target yang digunakan adalah Outcome, dimana 0 berarti tidak diabetes dan 1 berarti menderita diabetes.
# 
# 
# 

# %% [markdown]
# #### 2.2 Deskripsi Data
# 

# %%
#sample data
df.head()

# %%
#shape data
print(df.shape)

# %%
# data type
df.info()

# %%
# check null
display(df.isnull().sum())

# %%
# check duplicate
print('jumlah data duplikat : ',df.duplicated().sum())

# %%
# check deskripsi statistik
df.describe()

# %% [markdown]
# Tabel di atas menunjukkan hasil statistik dari data pasien pada 768 pasien wanita yang telah dites untuk diabetes. Beberapa hal menarik yang dapat dilihat dari hasil statistik tersebut antara lain:
# 
# - **Usia:** Rata-rata usia pasien adalah 33 tahun, dengan rentang usia antara 21-81 tahun. Usia merupakan faktor risiko penting pada diabetes, di mana semakin tua usia seseorang, semakin besar kemungkinan untuk terkena diabetes.
# - **Kehamilan:** Rata-rata jumlah kehamilan pada pasien adalah 4 kali (dibulatkan). Kehamilan yang berulang-ulang dapat meningkatkan risiko diabetes gestasional, yaitu diabetes yang terjadi selama kehamilan.
# - **Glukosa darah:** Rata-rata kadar glukosa dalam darah pasien adalah 120,8 mg/dL. Glukosa darah yang tinggi dapat menunjukkan adanya diabetes atau kondisi yang disebut prediabetes, di mana kadar glukosa darah lebih tinggi dari normal namun belum cukup tinggi untuk dianggap sebagai diabetes.
# - **Tekanan darah sistolik:** Rata-rata tekanan darah sistolik pada pasien adalah 69 mmHg. Tekanan darah yang tinggi dapat merusak pembuluh darah dan organ tubuh lainnya, dan dapat meningkatkan risiko diabetes.
# - **Ketebalan kulit:** Rata-rata ketebalan kulit pada pasien adalah 20,5 mm. Ketebalan kulit yang rendah dapat menjadi faktor risiko diabetes tipe 2, di mana tubuh tidak dapat menggunakan insulin dengan efektif.
# - **Insulin:** Rata-rata tingkat insulin pada pasien adalah 79,8 mu/L. Insulin yang tinggi dapat menunjukkan adanya resistensi insulin atau diabetes tipe 2.
# - **Indeks massa tubuh (BMI):** Rata-rata indeks massa tubuh (BMI) pada pasien adalah 31,9 kg/m2. BMI yang tinggi dapat menjadi faktor risiko diabetes tipe 2.
# - **DiabetesPedigreeFunction:** Rata-rata nilai fungsi DiabetesPedigreeFunction pada pasien adalah 0,47. Fungsi DiabetesPedigreeFunction menghitung risiko genetik seseorang terhadap diabetes tipe 2 berdasarkan riwayat keluarga.
# - **Outcome:** Sekitar 34,8% dari pasien terdiagnosis mengidap diabetes. Hal ini menunjukkan bahwa kebanyakan pasien pada dataset ini tidak menderita diabetes.

# %% [markdown]
# #### 2.3 Visualisasi Data

# %% [markdown]
# ##### 2.3.1 Presentasi Pasien Diabetes

# %%
plt.figure(figsize = (10,6))
colors = ['green', 'red']
labels = ['Tidak Diabetes', 'Diabetes']
plt.pie(df.Outcome.value_counts(), colors = colors, labels = labels, autopct = '%.1f%%', startangle = 90)
plt.title('Persentase Pasien Diabetes', fontweight = 'bold', fontsize = 18)
plt.ylabel('')
plt.style.use('seaborn-dark')
plt.show()



# %% [markdown]
# terdapat 65.1% pasien yang tidak menderita diabetes dan 34.9% pasien yang menderita diabetes.

# %% [markdown]
# ##### 2.3.2 Korelasi Fitur

# %%
corr = df.corr()
fig, ax = plt.subplots(figsize = (10,10))
cmap = sns.diverging_palette(230, 20, as_cmap = True)
mask = np.triu(np.ones_like(corr, dtype = bool))
sns.heatmap(corr, square = True, annot = True, linewidths = 1, cmap = cmap, mask = mask)

# %% [markdown]
# Tabel korelasi berikut menunjukkan korelasi antara beberapa variabel yang terdapat dalam dataset diabetes. Setiap sel pada tabel menunjukkan korelasi antara dua variabel. Nilai korelasi berkisar antara -1 hingga 1, dimana nilai -1 menunjukkan korelasi negatif sempurna, nilai 0 menunjukkan tidak adanya korelasi, dan nilai 1 menunjukkan korelasi positif sempurna.
# 
# Dari tabel ini didapatkan informasi bahwa tekanan darah dan ketebalan kulit tidak memiliki hubungan signifikan dengan diabetes sehingga dapat didrop dari dataset

# %% [markdown]
# ##### 2.3.3 Outier Detection

# %%
plt.figure(figsize = (10,10))
for i in range(0, len(df.columns)-1):
	plt.subplot(4,4,i+1)
	sns.boxplot(df.iloc[:,i])
	plt.title(df.columns[i], fontweight = 'bold', fontsize = 15)
	plt.xticks([])

# %% [markdown]
# hampir pada semua fitur memiliki outlier yang perlu dihapuskan

# %% [markdown]
# ### 3. Preprocessing Data

# %% [markdown]
# #### 3.1 Membuang kolom yang tidak diperlukan

# %%
drop_col = ['SkinThickness','BloodPressure']
df.drop(drop_col, axis = 1, inplace = True)
df.head()

# %% [markdown]
# #### 3.2 Menghapus Outlier

# %%
from sklearn.ensemble import IsolationForest
clf = IsolationForest(random_state = 0, contamination = 0.05)
clf.fit(df)
pred = clf.predict(df)
pred = pd.DataFrame(pred, columns = ['not_outlier'])
df=df.iloc[pred[pred['not_outlier'] == 1].index.values].reset_index(drop = True)
df

# %% [markdown]
# #### 3.3 data splitting

# %%
from sklearn.model_selection import train_test_split
X = df.drop('Outcome', axis = 1)
y = df['Outcome']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 101, stratify = y)
print('Jumlah baris dan kolom dari x_train adalah :', X_train.shape,', sedangkan Jumlah baris dan kolom dari y_train adalah :', y_train.shape)
print('Prosentase diabetes di y_train adalah :')
print(y_train.value_counts(normalize = True))
print('')
print('Jumlah baris dan kolom dari x_test adalah :', X_test.shape,', sedangkan Jumlah baris dan kolom dari y_test adalah :', y_test.shape)
print('Prosentase diabetes di y_test adalah :')
print(y_test.value_counts(normalize = True))




# %% [markdown]
# ##### 3.4 Standard Scaler

# %%
# standarisasi
scaler = StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)

# %% [markdown]
# ### 4. Modeling Dengan Tuning

# %% [markdown]
# #### 4.1 Logistic Regression

# %%
# parameter 
param_grid = {'penalty': ['l1', 'l2', 'elasticnet', 'none'],
              'C': [0.01, 0.1, 1, 10, 100],
              'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
              'max_iter': [100, 500, 1000]}
# model
model_lr = LogisticRegression()
# grid search
gscv = GridSearchCV(model_lr, param_grid=param_grid, scoring='recall', cv=10)
gscv.fit(X_train, y_train)

# %%
def evaluate_model(gscv, X_test, y_test):
    print('best parameter:')
    print(gscv.best_params_)

    print('\nbest cross validatiaon recall score:')
    print(gscv.best_score_)

    print('\nbest estimator:')
    model_lr = gscv.best_estimator_

    y_pred = model_lr.predict(X_test)
    print('\nclassification report:')
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    print('\nconfusion matrix:')
    sns.heatmap(cm, annot=True, cmap='Blues')
    plt.show()

evaluate_model(gscv, X_test, y_test)



# %% [markdown]
# #### 4.2 Random Forest

# %%
param_grid = {
    'n_estimators': [10, 50, 100, 200, 500],
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 10, 20],
}

# model
model_rf = RandomForestClassifier()
# grid search
gscv = GridSearchCV(model_rf, param_grid=param_grid, scoring='f1', cv=10)
gscv.fit(X_train, y_train)


# %%
evaluate_model(gscv, X_test, y_test)

# %% [markdown]
# ### 5 Modelling Tanpa Tuning

# %% [markdown]
# #### 5.1 Logistic Regression

# %%
model_lr = LogisticRegression()
model_lr.fit(X_train, y_train)

# %%
def evaluate_model_no_tune(model, X_test, y_test):


    y_pred = model.predict(X_test)
    print('\nclassification report:')
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    print('\nconfusion matrix:')
    sns.heatmap(cm, annot=True, cmap='Blues')
    plt.show()

# %%
evaluate_model_no_tune(model_lr, X_test, y_test)

# %% [markdown]
# #### 5.2 Random Forest

# %%
rf = RandomForestClassifier()
rf.fit(X_train, y_train)


# %%
evaluate_model_no_tune(rf, X_test, y_test)

# %% [markdown]
# ### 6. Hasil dan Kesimpulan

# %%
#note lr = LogisticRegression, rf = RandomForestClassifier, sebelum = sebelum tuning, sesudah = sesudah tuning

result={
    'lr sebelum':[0.52,0.62],
    'rf sebelum':[0.52,0.58],
    'lr sesudah':[0.80,0.73],
    'rf sesudah':[0.56,0.62]
}


result=pd.DataFrame(result,index=['recall','f1'])

result.plot(kind='bar',figsize=(10,10))
plt.title('Perbandingan hasil model')
plt.xticks(rotation=0)
plt.show()



# %% [markdown]
# dari hasil tersebut didapatkan bahwa model terbaik yang dihasilkan adalah logistic regression yang telah dituning dengan nilai recall sebesar 0.80 dan f1 score sebesar 0.73 dimana nilai recall dan f1 score tertinggi yang didapatkan pada penelitian sebelumnya [2](https://www.ijitee.org/wp-content/uploads/papers/v8i11/K21550981119.pdf) adalah 0.76 dan 0.75. Hal ini menunjukkan bahwa model yang dihasilkan telah mengalami peningkatan performansi pada recall namun penurunan pada f1 score. Selain itu juga didapat informasi bahwa hyperparameter tuning memiliki pengaruh untuk meningkatkan performansi model serta model random forest dinilai menghasilkan performansi yang lebih buruk dibandingkan dengan logistic regression.


