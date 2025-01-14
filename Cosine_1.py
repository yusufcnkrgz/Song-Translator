from resemblyzer import VoiceEncoder, preprocess_wav
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display

# VoiceEncoder ile embedding çıkarma ve benzerlik hesaplama
def calculate_similarity(file1, file2):
    encoder = VoiceEncoder()
    emb1 = encoder.embed_utterance(preprocess_wav(file1))
    emb2 = encoder.embed_utterance(preprocess_wav(file2))

    # Kosinüs benzerliğini hesapla
    similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
    return similarity

# Mel spektrogramı çizdirme
def plot_mel_spectrogram(audio, sr, title, ax):
    S = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128, fmax=8000)
    S_dB = librosa.power_to_db(S, ref=np.max)

    # Spektrogramı çizdir
    img = librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel', fmax=8000, ax=ax)
    ax.set_title(title)
    return img

# Ana fonksiyon
def process_audio(file1, file2):
    # Benzerlik hesapla
    similarity = calculate_similarity(file1, file2)
    print(f"Kosinüs Benzerliği (Resemblyzer): {similarity}")

    if similarity > 0.85:
        print("Bu ses dosyaları muhtemelen aynı kişi tarafından söylenmiştir.")
    else:
        print("Bu ses dosyaları farklı kişiler tarafından söylenmiş olabilir.")

    # Ses dosyalarını yükleme
    y1, sr1 = librosa.load(file1, sr=None)
    y2, sr2 = librosa.load(file2, sr=None)

    # Her iki dosyayı kısa olana göre hizalama
    min_length = min(len(y1), len(y2))
    y1 = y1[:min_length]
    y2 = y2[:min_length]

    # Mel spektrogramları çizdir
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    img1 = plot_mel_spectrogram(y1, sr1, "Original (File1)", axes[0])
    img2 = plot_mel_spectrogram(y2, sr2, "Converted (File2)", axes[1])

    # Ortak renk çubuğu ekleme
    plt.subplots_adjust(right=0.85)  # Grafiğin sağ kenarına daha fazla boşluk bırakır
    cbar = fig.colorbar(img1, ax=axes[1], orientation='vertical', fraction=0.05, pad=0.05, location='right')


    # PNG olarak kaydetme
    plt.tight_layout()
    plt.savefig("mel_spectrograms_with_similarity.png")
    plt.show()

    return similarity

# Ses dosyalarının yollarını belirtin
file1 = "Original.mp3"
file2 = "Converted.mp3"

# İşlem başlat
similarity = process_audio(file1, file2)
