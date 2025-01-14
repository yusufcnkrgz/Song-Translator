from resemblyzer import VoiceEncoder, preprocess_wav
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

def calculate_similarity(file1, file2):
    """İki ses dosyası arasındaki benzerliği hesaplar."""
    encoder = VoiceEncoder()
    emb1 = encoder.embed_utterance(preprocess_wav(file1))
    emb2 = encoder.embed_utterance(preprocess_wav(file2))

    # Kosinüs benzerliğini hesapla
    similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
    return similarity

def find_most_similar(input_file, reference_files):
    """Input ses dosyasını referans dosyalarla karşılaştırır ve en benzer olanı döndürür."""
    similarities = {}

    for artist, ref_file in reference_files.items():
        try:
            similarity = calculate_similarity(input_file, ref_file)
            similarities[artist] = similarity
            print(f"{artist}: Benzerlik = {similarity:.2f}")
        except Exception as e:
            print(f"{artist}: Hata oluştu: {e}")

    # En yüksek benzerliğe sahip olanı bulma
    if similarities:
        most_similar_artist = max(similarities, key=similarities.get)
        highest_similarity = similarities[most_similar_artist]
        return most_similar_artist, highest_similarity
    else:
        return None, 0

# Ana işlem fonksiyonu
def process_audio(input_file):
    """Verilen input dosyasını referans dosyalarla karşılaştırır."""
    # Referans dosyalar ve sanatçılar
    reference_files = {
        "Kazım Koyuncu (Hayde)": "hayde_kazım_koyuncu [vocals].mp3",
        "İbrahim Tatlıses (Mavisim)": "Ibrahim-Tatlises-Mavisim [vocals].mp3",
        "Kazım Koyuncu (Hayde 2)": "Kazim_Koyuncu_-Hayde-_www.BiG.AZ (Vocals) (MDX v2 Voc FT).mp3",
        "Sezen Aksu (Firuze)": "Sezen Aksu - Firuze (Official Audio - Orijinal Plak Kayıt) (Vocals) (MDX v2 Voc FT).mp3",
        "Tarkan (Şımarık)": "Tarkan_-Simarik-_wwwBiGAZ-vocals.mp3"
    }

    # En benzer sanatçıyı bulma
    most_similar_artist, highest_similarity = find_most_similar(input_file, reference_files)

    if most_similar_artist:
        print(f"\nEn benzer sanatçı: {most_similar_artist} (Benzerlik: {highest_similarity:.2f})")
    else:
        print("\nBenzer sanatçı bulunamadı veya dosyalar işlenemedi.")

    return most_similar_artist, highest_similarity

# Kullanıcıdan input dosyasını al
input_file = "input.mp3"  # Burada input dosyanızın yolunu belirtin

# İşlem başlat
most_similar_artist, similarity = process_audio(input_file)
