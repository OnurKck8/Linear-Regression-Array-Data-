#!/usr/bin/env python
# coding: utf-8

# In[65]:


# Arrayler ile çalıştığımız için
import numpy as np
# Doğrulama yapabilmek için Import ettim
from sklearn.model_selection import train_test_split
# Doğrusal regresyon yapabilmek için linear kütüphanesi import ettim
from sklearn.linear_model import LinearRegression
# Görselleştirmek için kütuphane
import matplotlib.pyplot as plt
# R2 Skoruna bakabilmek için import ettim
from sklearn.metrics import r2_score


# In[66]:


revenue = np.array([6000, 9000, 12000, 15000, 18000, 4500, 7500, 12000, 9000, 16500, 10500, 18000, 6000, 13500, 12000, 19500, 7500, 15000, 9000, 16500, 4500, 12000, 10500, 18000, 6000, 13500, 7500, 15000, 9000, 16500])
investment = np.array([2000, 3000, 4000, 5000, 6000, 1500, 2500, 4000, 3000, 5500, 3500, 6000, 2000, 4500, 4000, 6500, 2500, 5000, 3000, 5500, 1500, 4000, 3500, 6000, 2000, 4500, 2500, 5000, 3000, 5500])


# In[67]:


# Bağımlı ve bağımsız değişkenleri tanımlayın
X = revenue.reshape(-1,1) #Matrix yapmamız lazım
y = investment.reshape(-1,1)

# Veri setini eğitim ve test setlerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) # %80 eğit  %20 tahmin et

# Lineer regresyon modeli oluşturma ve eğitme
model = LinearRegression()
model.fit(X_train, y_train)


# In[68]:


# Eğitim ve test veri setleri üzerinde tahmin yapma
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Eğitim ve test veri setleri için doğruluk oranı hesaplama
accuracy_train = model.score(X_train, y_train)
accuracy_test = model.score(X_test, y_test)

r2_train = r2_score(X_train, y_train)
r2_test = r2_score(X_test, y_test)

print('Eğitim Seti Doğruluk Oranı:', accuracy_train)
print('Test Seti Doğruluk Oranı:', accuracy_test)
print('Eğitim Seti R² Değeri:', r2_train)
print('Test Seti R² Değeri:', r2_test)


# In[69]:


#Neyi tahmin edeceksin
yeni_yatirim = model.predict([[4000]])
print('Yatırım Alıp Almama Durumu: ', yeni_yatirim)


# In[70]:


# Görselleştirmeye lojistik regresyon eğrisini ekleyin
plt.scatter(revenue, investment, label='Gerçek Gelir')
plt.plot(revenue, model.predict(X), color='red', label='Tahmini Gelir')
plt.xlabel('Revenue')
plt.ylabel('Investment')
plt.title('Lineer Regresyon Modeli ile Tahmini Yatırım')
plt.legend()
plt.show()

