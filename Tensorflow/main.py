import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split

# Veri setini yükleme
data = pd.read_csv('dataset.csv', header=None)
print("Veri Şekli : ", data.shape)
print(data.head())

# Veri setini özellik matrisi ve etiket vektörü olarak ayırma
x_orig = data.iloc[:, [0, 1]].values
y_orig = data.iloc[:, -1].values
print("Özellik Matrisi Şekli : ", x_orig.shape)
print("Etiket Vektörü Şekli : ", y_orig.shape)

# Veriyi eğitim ve test setlerine ayırma
x_train, x_test, y_train, y_test = train_test_split(x_orig, y_orig, test_size=0.2, random_state=42)

# Normalizasyon
x_train = (x_train - np.mean(x_train, axis=0)) / np.std(x_train, axis=0)
x_test = (x_test - np.mean(x_test, axis=0)) / np.std(x_test, axis=0)

# Hyperparameters
alpha, epochs = 0.0035, 500

# Modelin oluşturulması ve tanımlanması
model = tf.keras.Sequential([
    tf.keras.layers.Dense(1, activation='sigmoid', input_shape=(x_train.shape[1],))
])
model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])

# Modelin eğitilmesi
history = model.fit(x_train, y_train, epochs=epochs, verbose=1)

# Modelin ilk katmanının ağırlıklarını ve biasını almak için
weights, biases = model.layers[0].get_weights()

# Fonksiyonun tanımlanması
def plot_decision_boundary(X, y, model):
    # Modelin ilk katmanının ağırlıklarını ve biasını almak için
    weights, biases = model.layers[0].get_weights()

    # Veri setinin özelliklerine göre min ve max değerlerini ayarlayın
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    # Izgara oluşturma
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))

    # Modelin tahminlerini alın
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z > 0.5  # Sigmoid fonksiyonu için eşik değeri
    Z = Z.reshape(xx.shape)

    # Karar sınırı çizgisi
    plt.contour(xx, yy, Z, levels=[0.5], cmap="Paired")

    # Veri noktalarını çiz
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)

    # Eksenleri isimlendir
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')

    # Başlık ekle
    plt.title('Karar Sınırları')

    # Karar sınırı çizgisi için x2 değerleri
    w1, w2 = weights[0][0], weights[1][0]
    bias = biases[0]
    x_values = np.linspace(start=x_min, stop=x_max, num=100)
    y_values = -(w1 / w2) * x_values - (bias / w2)

    # Karar sınırı çizgisini çiz
    plt.plot(x_values, y_values, 'k--')

    # Grafiği göster
    plt.show()

# Karar sınırlarını görselleştirmek için fonksiyonu çağırma
plot_decision_boundary(x_orig, y_orig, model)

# Veriyi görselleştirme
x_pos = x_orig[y_orig == 1]
x_neg = x_orig[y_orig == 0]

plt.scatter(x_pos[:, 0], x_pos[:, 1], color='green', label='Pozitif')
plt.scatter(x_neg[:, 0], x_neg[:, 1], color='red', label='Negatif')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Verilerin Görsel Hali')
plt.legend()
plt.show()

# Veriyi eğitim ve test setlerine ayırma
x_train, x_test, y_train, y_test = train_test_split(x_orig, y_orig, test_size=0.2, random_state=42)

# Normalizasyon
x_train = (x_train - np.mean(x_train, axis=0)) / np.std(x_train, axis=0)
x_test = (x_test - np.mean(x_test, axis=0)) / np.std(x_test, axis=0)

# Hyperparameters
alpha, epochs = 0.0035, 500

# Modelin oluşturulması ve tanımlanması
model = tf.keras.Sequential([
    tf.keras.layers.Dense(1, activation='sigmoid', input_shape=(x_train.shape[1],))
])
model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])

# Modelin eğitilmesi
history = model.fit(x_train, y_train, epochs=epochs, verbose=1)

# Epochs-Cost grafiğini oluşturma
# Modelin eğitim sırasında elde edilen maliyet değerlerini alın
train_loss = history.history['loss']

# Yineleme sayısına karşılık gelen indeks değerlerini oluşturun
iterations = range(1, len(train_loss) + 1)

# Maliyet grafiğini çizin
plt.plot(iterations, train_loss, label='Eğitim Maliyeti')
plt.xlabel('Yineleme Sayısı')
plt.ylabel('Maliyet')
plt.title('Yineleme-Maliyet Grafiği')
plt.legend()
plt.show()

# Kayıp grafiğini çiz
plt.plot(iterations, history.history['loss'])
plt.xlabel('Yineleme')
plt.ylabel('Kayıp')
plt.title('Yineleme-Kayıp Grafiği')
plt.show()

# Doğruluk grafiğini çiz
plt.plot(iterations, history.history['accuracy'])
plt.xlabel('Yineleme')
plt.ylabel('Doğruluk')
plt.title('Yineleme-Doğruluk Grafiği')
plt.show()

# Test seti üzerinde modelin değerlendirilmesi
loss, accuracy = model.evaluate(x_test, y_test)
print("Test Loss:", loss)
print("Test Accuracy:", accuracy)

# Eğitim sonrası modelin kullanılması
predictions = model.predict(x_test)

# Karar sınırlarını görselleştirmek için fonksiyonu çağırma
plot_decision_boundary(x_orig, y_orig, model)

# Modelin ilk katmanının ağırlıklarını ve biasını almak için
weights, biases = model.layers[0].get_weights()

# Ağırlıkları ve biası yazdır
print("Ağırlıklar:", weights)
print("Bias:", biases)




















