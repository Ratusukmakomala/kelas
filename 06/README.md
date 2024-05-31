# ANN
Kerjakan:

1. Jelaskan beberapa contoh fungsi aktivasi

Fungsi aktivasi adalah komponen penting dalam jaringan saraf tiruan (neural network) yang memperkenalkan non- neuralitas ke dalam model, memungkinkan jaringan untuk belajar dan mempresentasikan data yang lebih kompleks. Berikut beberapa fungsi aktivasi:

- sigmoid

 ![image](https://github.com/Ratusukmakomala/kelas/assets/92583035/53bb087c-1b61-4760-9e34-0d7d4d5a3b72)

  Secara matematis dapat direpresentasikan sebagai:

![image](https://github.com/Ratusukmakomala/kelas/assets/92583035/8a211a98-37e7-47db-a477-407f2af55ff7)

Fungsi sigmoid adalah AF non-linier yang digunakan terutama dalam jaringan saraf feedforward. Ini adalah fungsi nyata yang dapat terdiferensiasi, didefinisikan untuk nilai masukan nyata, dan mengandung turunan positif di mana pun dengan tingkat kehalusan tertentu. Fungsi sigmoid muncul di lapisan keluaran model pembelajaran mendalam dan digunakan untuk memprediksi keluaran berbasis probabilitas.

kelebihan : memetakan input menjadi rentang antara 0 dan 1 sehingga cocok untuk model probabilitas.

Kekurangan: Masalah vanishing gradient pada lapisan yang lebih dalam, yang membuat pelatihan jaringan menjadi sulit.

Rumus matematika untuk fungsi aktivasi ini adalah:

- Tan h (Hyperbolic Tangent)

  ![image](https://github.com/Ratusukmakomala/kelas/assets/92583035/75a45cd0-07de-453b-8961-203e34be268c)


  Secara matematis dapat direpresentasikan sebagai:

  ![image](https://github.com/Ratusukmakomala/kelas/assets/92583035/c56bc412-7ba8-4089-b5e2-3d07f5e9107a)

Fungsi tangen hiperbolik alias fungsi tanh adalah jenis AF lainnya. Ini adalah fungsi yang lebih halus dan berpusat pada nol yang memiliki rentang antara -1 hingga 1. 

Kelebihan: Mmemetakan input menjadi rentang antara -1 dan 1, yang seringkali menghasilkan konvergensi lebih cepat daripada sigmoid.

Kekurangan: Sama seperti sigmoid, tan h juga dapat mengalami vanishing gradient.

- ReLu (Rectified Linear Unit)
  
![image](https://github.com/Ratusukmakomala/kelas/assets/92583035/88f796f5-969b-4d9f-8d08-14c75616fac7)

Secara matematis dapat direpresentasikan sebagai:

![image](https://github.com/Ratusukmakomala/kelas/assets/92583035/124e3bb4-8353-4a38-8d03-fbef7e544238)

fungsi rectified linear unit (ReLU), adalah AF belajar cepat yang menjanjikan performa canggih dengan hasil luar biasa. Dibandingkan dengan AF lain seperti fungsi sigmoid dan tanh, fungsi ReLU menawarkan performa dan generalisasi yang jauh lebih baik dalam pembelajaran mendalam. Fungsi tersebut merupakan fungsi hampir linier yang mempertahankan properti model linier, sehingga mudah dioptimalkan dengan metode penurunan gradien. 

Kelebihan: Mengatasi masalah vanishing gradient dengan baik dan sangat populer, dikarenakan konvergensinya cepat.

Kekurangan: Dapat menyebabkan "dead neurons" dimana neuron berhenti memperbarui karena gradien nol.

- Leaky ReLu 

![image](https://github.com/Ratusukmakomala/kelas/assets/92583035/14dbb6f4-60ba-4afb-9729-24f4ab891b9d)

Secara matematis dapat direpresentasikan sebagai:

![image](https://github.com/Ratusukmakomala/kelas/assets/92583035/d9aad30b-6761-49ca-a5dd-f7c15e383bce)

Leaky ReLU merupakan versi perbaikan dari fungsi ReLU untuk mengatasi masalah Dying ReLU karena memiliki kemiringan positif yang kecil di area negatif.

Kelebihan: Mengatasi masalah "dead neuros" dengan memberikan gradien kecil untuk nilai negatif.

Kekurangan: Perlu memilih konstanta leak yang optimal, biasanya dipilih secara ad-hoc

- Softmax

![image](https://github.com/Ratusukmakomala/kelas/assets/92583035/3ffe4874-c13a-4f2d-9dda-50b85c98759e)

![image](https://github.com/Ratusukmakomala/kelas/assets/92583035/c1cb3ebf-a937-4a8d-ba31-6123a2f16cee)

Fungsi softmax adalah jenis AF lain yang digunakan dalam jaringan saraf untuk menghitung distribusi probabilitas dari vektor bilangan real. Fungsi ini menghasilkan keluaran yang berkisar antara nilai 0 dan 1 dan dengan jumlah probabilitas sama dengan.

Kelebihan: Mengubah output menjadi distribusi probabilitas yang totalnya 1, sering digunakan pada lapisan output untuk klasifikasi multi kelas.

Kekurangan: Dapat mengalami masalah numeric overflow atau underflow pada input yang sangat besar atau sangat kecil.

- Swish

![image](https://github.com/Ratusukmakomala/kelas/assets/92583035/9ea9240e-2aaf-4742-9b33-2a776048f2e6)


Secara matematis dapat direpresentasikan sebagai:

![image](https://github.com/Ratusukmakomala/kelas/assets/92583035/abce9d0b-7147-44d0-a4b0-8f0c090a7d14)

Swish adalah fungsi halus yang artinya tidak berubah arah secara tiba-tiba seperti yang dilakukan ReLU di dekat x = 0. Sebaliknya, fungsi tersebut secara mulus membengkok dari 0 menuju nilai < 0 dan kemudian naik lagi. Swish secara konsisten mencocokkan atau mengungguli fungsi aktivasi ReLU pada jaringan dalam yang diterapkan pada berbagai domain menantang seperti klasifikasi gambar , terjemahan mesin. 

Kelebihan: Ditemukan secara otomatis melalui pencarian arsitektur neural, sering menghasilkan performa yang lebih baik dibandingkan ReLu dan fungsi aktivasi lainnya pada beberapa tugas.

Kekurangan: Lebih Kompleks dalam perhitungan dibandingkan dengan ReLu, namun seringkali sepadan dengan peningkatan performa.


2. Coba gambarkan dan jelaskan beberapa arsitektur berikut:
   
* GAN

  ![image](https://github.com/Ratusukmakomala/kelas/assets/92583035/294d9780-37d5-40ed-9bf1-e1f0b2a2e2ab)

Generative Adversarial Network (GAN) adalah jenis arsitektur jaringan saraf tiruan yang terdiri dari dua bagian utama: generator dan discriminator. Keduanya beroperasi dalam suatu siklus pelatihan yang iteratif, di mana generator mencoba membuat data semirip mungkin dengan data nyata, sementara discriminator berusaha membedakan antara data yang dihasilkan oleh generator dan data nyata.

Fungsi Generative Adversarial Network (GAN):
a. Pembuatan Data Baru:

GAN digunakan untuk menghasilkan data baru yang serupa dengan data latihannya. Contoh aplikasinya termasuk pembuatan gambar, musik, atau teks baru.

b. Peningkatan Kualitas Gambar:

Dalam bidang pengolahan gambar, GAN dapat digunakan untuk meningkatkan resolusi dan kualitas gambar, memberikan hasil yang lebih realistis.

c. Transfer Gaya:

GAN memungkinkan transfer gaya dari satu gambar ke gambar lain, memberikan elemen estetika dari satu gambar ke gambar lainnya.

d. Augmentasi Data:

GAN dapat digunakan untuk membuat data tambahan yang dapat digunakan dalam pelatihan model machine learning, membantu meningkatkan kinerja model.

e. Generasi Wajah dan Orang Palsu (Deepfake):

GAN digunakan untuk menciptakan wajah atau video orang yang tampak sangat nyata, yang dapat menimbulkan tantangan etika dalam konteks deepfake.

Kelebihan Generative Adversarial Network (GAN):

a. Fleksibilitas dalam Pembuatan Data

b. Pemahaman Konten dan Stil

c. Pembelajaran Tanpa Pengawasan

d. Peningkatan Kreativitas

e. Aplikasi di Berbagai Bidang

* AE

![WhatsApp Image 2024-05-31 at 12 33 00_85c6e292](https://github.com/Ratusukmakomala/kelas/assets/92583035/1dff1055-d0ce-4fc0-b9e7-9b28034de678)

Autoencoder adalah jenis arsitektur jaringan saraf yang dirancang untuk secara efisien mengompresi (encode) data input hingga ke fitur-fitur esensialnya, kemudian merekonstruksi (decode) input asli dari representasi yang dikompresi ini.

Autoencoder vs. encoder-decoder
Meskipun semua model autoencoder menyertakan encoder dan decoder, namun tidak semua model encoder-decoder adalah autoencoder.

Bagaimana cara kerja autoencoder?

Autoencoder menemukan variabel laten dengan melewatkan data input melalui "kemacetan" sebelum mencapai decoder. Hal ini memaksa encoder untuk belajar mengekstrak dan melewatkan hanya informasi yang paling kondusif untuk merekonstruksi input asli secara akurat.

Contoh penggunaan autoencoder:

-Kompresi Data

-Pengurangan Dimensi

-Deteksi anomali dan pengenalan wajah

-Denoising gambar dan denoising audio

-Rekonstruksi gambar

-Tugas generatif

* LSTM

LSTM (Memori Jangka Pendek Panjang)
LSTM merupakan modifikasi dari RNN yang memiliki memori dan banyak jenis gerbang yaitu input gate, forget gate, dan output gate . LSTM mampu mempelajari lebih dari 1000 langkah sebelumnya tergantung pada kompleksitas jaringan.

![image](https://github.com/Ratusukmakomala/kelas/assets/92583035/6e617386-255b-4e12-8f96-89236ff7864b)

* VGG

VGG (Visual Geometry Group) adalah Jaringan Neural Konvolusional (CNN) dalam yang terdiri dari beberapa lapisan melahirkan Revolusi Pembelajaran Mendalam yang dipimpin oleh VGG-16 dan VGG-19. Secara khusus, “dalam” mengacu pada jumlah lapisan konvolusional di VGG-16 (16) atau VGG-19 (19).

Mari kita lihat sekilas arsitektur VGG:

-Masukan: Ukuran masukan gambar untuk VGGNet adalah 224 x 224 piksel. Selain itu, untuk kompetisi ImageNet, pembuat model menghapus patch tengah 224×224 di setiap gambar untuk menjaga konsistensi gambar masukan.

- Lapisan Konvolusional: Lapisan konvolusional VGG menggunakan bidang reseptif minimum 3 × 3 yang kecil, namun menggabungkan gerakan atas/bawah dan kiri/kanan. Selain itu, ini melibatkan transformasi linier dari masukan melalui filter konvolusi 1×1. Setelah itu hadirlah unit ReLU, sebuah penemuan yang berasal dari AlexNet dan ini membantu meminimalkan waktu pelatihan. Fungsi aktivasi unit linier yang diperbaiki dikenal sebagai ReLU.

-Lapisan Tersembunyi: Setiap lapisan tersembunyi VGG menggunakan ReLU. Normalisasi Respon Lokal (LRN) biasanya tidak dimanfaatkan oleh VGG karena mengakibatkan pemborosan memori dan waktu pelatihan. Selain itu, ini tidak meningkatkan akurasi keseluruhan sama sekali.

-Lapisan Terhubung Sepenuhnya: Ini, ada tiga lapisan yang terhubung sepenuhnya di VGGnet. Dua lapisan pertama masing-masing memiliki 4096 saluran, sedangkan lapisan ketiga memiliki 1 saluran per kelas — yang berarti total 1000 saluran.

Lapisan yang terhubung sepenuhnya:

![image](https://github.com/Ratusukmakomala/kelas/assets/92583035/07e96aeb-21a0-4566-a6fd-1d6b6fa41009)

Arsitektur VGG-16 :

![image](https://github.com/Ratusukmakomala/kelas/assets/92583035/6e701627-da61-4c55-8d39-02bad8b4c014)

Arsitektur Jaringan Neural Konvolusional: Data gambar adalah masukan dari CNN; keluaran model menyediakan kategori prediksi untuk gambar masukan:

![image](https://github.com/Ratusukmakomala/kelas/assets/92583035/65df0854-9011-434d-8a3f-4c7a5a9733f4)


Arsitektur Jaringan Neural VGG16:

![image](https://github.com/Ratusukmakomala/kelas/assets/92583035/75c5d560-383b-424a-8e54-2846c28f824c)


* RNN

  ![image](https://github.com/Ratusukmakomala/kelas/assets/92583035/e9edc619-2b12-4f27-b24d-db529c10c564)

RNN (Recurrent Neural Network) adalah salah satu jenis arsitektur ANN yang digunakan untuk memproses urutan data atau rangkaian, seperti teks, audio, atau data waktu. RNN memiliki kemampuan untuk mengingat informasi dari waktu sebelumnya dan menggunakan informasi itu untuk menghasilkan output pada waktu saat ini.

RNN memiliki kelemahan hanya mampu melihat sinyal sebelumnya kurang lebih 10 langkah. Hal ini disebabkan karena RNN memiliki masalah dalam mempelajari ketergantungan jarak jauh dalam urutan data karena sifat yang mengalami vanishing gradien. Akibatnya, RNN sering mengalami kesulitan dalam mengingat informasi dari jarak waktu yang lama dan hanya mampu mengingat informasi terbatas dari waktu sebelumnya. Kekurangan dari RNN ini sudah diperbaiki dengan terciptanya LSTM (Long Short-Term Memory).


3. Buatkan kodingan from scracth NN

PADA VISUAL STUDIO CODE BUAT FILE simple_nn.py

```
import numpy as np

# Fungsi aktivasi sigmoid dan turunannya
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Fungsi untuk melatih jaringan saraf
def train(X, y, hidden_neurons, epochs, learning_rate):
    input_neurons = X.shape[1]
    output_neurons = 1

    # Inisialisasi bobot secara acak
    weights_input_hidden = np.random.uniform(size=(input_neurons, hidden_neurons))
    weights_hidden_output = np.random.uniform(size=(hidden_neurons, output_neurons))

    # Proses pelatihan
    for _ in range(epochs):
        # Forward propagation
        hidden_layer_activation = np.dot(X, weights_input_hidden)
        hidden_layer_output = sigmoid(hidden_layer_activation)

        output_layer_activation = np.dot(hidden_layer_output, weights_hidden_output)
        predicted_output = sigmoid(output_layer_activation)

        # Backpropagation
        error = y - predicted_output
        d_predicted_output = error * sigmoid_derivative(predicted_output)

        error_hidden_layer = d_predicted_output.dot(weights_hidden_output.T)
        d_hidden_layer = error_hidden_layer * sigmoid_derivative(hidden_layer_output)

        # Update bobot
        weights_hidden_output += hidden_layer_output.T.dot(d_predicted_output) * learning_rate
        weights_input_hidden += X.T.dot(d_hidden_layer) * learning_rate

    return weights_input_hidden, weights_hidden_output

# Fungsi untuk memprediksi menggunakan jaringan saraf yang sudah dilatih
def predict(X, weights_input_hidden, weights_hidden_output):
    hidden_layer_activation = np.dot(X, weights_input_hidden)
    hidden_layer_output = sigmoid(hidden_layer_activation)

    output_layer_activation = np.dot(hidden_layer_output, weights_hidden_output)
    predicted_output = sigmoid(output_layer_activation)

    return predicted_output

# Contoh penggunaan
if __name__ == "__main__":
    # Data pelatihan (XOR problem)
    X = np.array([[0, 0],
                  [0, 1],
                  [1, 0],
                  [1, 1]])
    y = np.array([[0], [1], [1], [0]])

    # Parameter jaringan
    hidden_neurons = 2
    epochs = 10000
    learning_rate = 0.1

    # Pelatihan
    weights_input_hidden, weights_hidden_output = train(X, y, hidden_neurons, epochs, learning_rate)

    # Prediksi
    predictions = predict(X, weights_input_hidden, weights_hidden_output)
    print("Prediksi:")
    print(predictions)
```
    Hasilnya running pada code diatas adalah:

   ![WhatsApp Image 2024-05-31 at 12 59 04_026da93b](https://github.com/Ratusukmakomala/kelas/assets/92583035/f9c6bbdf-45f1-4915-81e1-635a8d948320)



