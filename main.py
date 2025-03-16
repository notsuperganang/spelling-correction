import re
from collections import Counter
import time
import numpy as np
import os
from datetime import datetime
import io
import sys

class OutputCapture:
    def __init__(self):
        self.log_data = []
        self.original_stdout = sys.stdout
        self.output_buffer = io.StringIO()
    
    def start_capture(self):
        class DualWriter:
            def __init__(self, buffer, original_stdout):
                self.buffer = buffer
                self.original_stdout = original_stdout
            
            def write(self, text):
                self.buffer.write(text)
                self.original_stdout.write(text)
            
            def flush(self):
                self.original_stdout.flush()
        
        sys.stdout = DualWriter(self.output_buffer, self.original_stdout)
    
    def stop_capture(self):
        sys.stdout = self.original_stdout
    
    def get_output(self):
        return self.output_buffer.getvalue()
    
    def save_to_file(self, filename):
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(self.get_output())
        print(f"Hasil evaluasi disimpan ke dalam file: {filename}")

# ==================== PEMROSESAN KORPUS ====================
def process_corpus(file_path):
    """
    Membaca dan memproses korpus teks untuk membuat daftar kata dan frekuensinya.
    
    Args:
        file_path (str): Path ke file korpus
        
    Returns:
        Counter: Frekuensi kemunculan setiap kata dalam korpus
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read().lower()
    except UnicodeDecodeError:
        # Mencoba encoding lain jika utf-8 gagal
        with open(file_path, 'r', encoding='latin-1') as file:
            text = file.read().lower()
            
    # Ekstrak semua kata dengan mempertahankan huruf, angka, dan apostrof
    words = re.findall(r'\w+', text)
    return Counter(words), text

def build_vocabulary(word_counts, min_frequency=1):
    """
    Membangun kamus kata yang valid berdasarkan frekuensi kemunculannya.
    
    Args:
        word_counts (Counter): Counter berisi frekuensi kata
        min_frequency (int): Frekuensi minimum untuk kata yang akan dimasukkan dalam kamus
        
    Returns:
        set: Kumpulan kata yang valid
    """
    return {word for word, count in word_counts.items() if count >= min_frequency}

def compute_ngram_frequencies(corpus, n=2):
    """
    Menghitung frekuensi n-gram dalam korpus.
    
    Args:
        corpus (str): Teks korpus
        n (int): Ukuran n-gram
        
    Returns:
        Counter: Frekuensi n-gram
    """
    words = re.findall(r'\w+', corpus.lower())
    ngrams = []
    
    for i in range(len(words) - n + 1):
        ngram = tuple(words[i:i+n])
        ngrams.append(ngram)
    
    return Counter(ngrams)

# ==================== ALGORITMA LEVENSHTEIN ====================
def levenshtein_distance(s1, s2):
    """
    Menghitung jarak Levenshtein antara dua string.
    
    Args:
        s1 (str): String pertama
        s2 (str): String kedua
        
    Returns:
        int: Jarak edit antara s1 dan s2
    """
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    
    # s1 sekarang adalah string yang lebih panjang
    if len(s2) == 0:
        return len(s1)
    
    # Inisialisasi matriks
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            # Hitung biaya operasi: insert, delete, atau substitute
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            
            # Ambil biaya minimum
            current_row.append(min(insertions, deletions, substitutions))
        
        previous_row = current_row
    
    return previous_row[-1]

def levenshtein_distance_matrix(s1, s2):
    """
    Versi Levenshtein Distance yang mengembalikan seluruh matriks jarak.
    Berguna untuk debugging dan visualisasi.
    
    Args:
        s1 (str): String pertama
        s2 (str): String kedua
        
    Returns:
        numpy.ndarray: Matriks jarak edit
    """
    # Inisialisasi matriks dengan dimensi (len(s1)+1) x (len(s2)+1)
    matrix = np.zeros((len(s1) + 1, len(s2) + 1), dtype=int)
    
    # Set nilai baris pertama
    for i in range(len(s1) + 1):
        matrix[i, 0] = i
        
    # Set nilai kolom pertama
    for j in range(len(s2) + 1):
        matrix[0, j] = j
    
    # Isi matriks
    for i in range(1, len(s1) + 1):
        for j in range(1, len(s2) + 1):
            if s1[i-1] == s2[j-1]:
                matrix[i, j] = matrix[i-1, j-1]  # Karakter sama, tidak ada biaya
            else:
                # Pilih operasi dengan biaya minimum
                matrix[i, j] = min(
                    matrix[i-1, j] + 1,      # Delete
                    matrix[i, j-1] + 1,      # Insert
                    matrix[i-1, j-1] + 1     # Substitute
                )
    
    return matrix

# ==================== KANDIDAT KOREKSI ====================
def generate_candidates_edit_distance_1(word):
    """
    Menghasilkan semua kemungkinan kata dengan jarak edit 1 dari kata asli.
    
    Args:
        word (str): Kata asli
        
    Returns:
        set: Semua kandidat kata dengan jarak edit 1
    """
    letters = 'abcdefghijklmnopqrstuvwxyz'
    splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    
    # Berbagai operasi edit
    deletes = [L + R[1:] for L, R in splits if R]  # Hapus satu karakter
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]  # Tukar dua karakter
    replaces = [L + c + R[1:] for L, R in splits if R for c in letters]  # Ganti satu karakter
    inserts = [L + c + R for L, R in splits for c in letters]  # Sisipkan satu karakter
    
    return set(deletes + transposes + replaces + inserts)

def generate_candidates_edit_distance_2(word):
    """
    Menghasilkan semua kemungkinan kata dengan jarak edit 2 dari kata asli.
    
    Args:
        word (str): Kata asli
        
    Returns:
        set: Semua kandidat kata dengan jarak edit 2
    """
    return {e2 for e1 in generate_candidates_edit_distance_1(word) 
           for e2 in generate_candidates_edit_distance_1(e1)}

def filter_known_words(candidates, vocabulary):
    """
    Menyaring kandidat kata untuk hanya menyertakan kata yang ada dalam kamus.
    
    Args:
        candidates (set): Kumpulan kandidat kata
        vocabulary (set): Kamus kata yang valid
        
    Returns:
        set: Kandidat kata yang ada dalam kamus
    """
    return {word for word in candidates if word in vocabulary}

# ==================== ALGORITMA KOREKSI EJAAN ====================
def correct_word_levenshtein(word, word_counts, vocabulary):
    """
    Memperbaiki kata yang salah eja menggunakan jarak Levenshtein.
    
    Args:
        word (str): Kata yang mungkin salah eja
        word_counts (Counter): Frekuensi kata dalam korpus
        vocabulary (set): Kumpulan kata yang valid
        
    Returns:
        str: Saran koreksi ejaan
    """
    # Jika kata sudah ada dalam kamus, kembalikan kata tersebut
    if word in vocabulary:
        return word
    
    # Cari kandidat dengan jarak edit 1 dan 2
    candidates_edit1 = filter_known_words(generate_candidates_edit_distance_1(word), vocabulary)
    if candidates_edit1:
        return max(candidates_edit1, key=lambda w: word_counts[w])
    
    candidates_edit2 = filter_known_words(generate_candidates_edit_distance_2(word), vocabulary)
    if candidates_edit2:
        return max(candidates_edit2, key=lambda w: word_counts[w])
    
    # Jika tidak ada kandidat yang cocok, kembalikan kata asli
    return word

def predict_with_context(misspelled_word, prev_word, next_word, word_counts, vocabulary, bigram_counts):
    """
    Memperbaiki ejaan dengan mempertimbangkan konteks kata.
    
    Args:
        misspelled_word (str): Kata yang mungkin salah eja
        prev_word (str): Kata sebelumnya
        next_word (str): Kata setelahnya
        word_counts (Counter): Frekuensi kata dalam korpus
        vocabulary (set): Kumpulan kata yang valid
        bigram_counts (Counter): Frekuensi bigram
        
    Returns:
        str: Saran koreksi ejaan
    """
    candidates = []
    
    # Jika kata sudah benar, kembalikan
    if misspelled_word in vocabulary:
        return misspelled_word
    
    # Dapatkan kandidat dengan jarak edit 1
    candidates_edit1 = filter_known_words(generate_candidates_edit_distance_1(misspelled_word), vocabulary)
    
    # Jika tidak ada, coba jarak edit 2
    if not candidates_edit1:
        candidates_edit1 = filter_known_words(generate_candidates_edit_distance_2(misspelled_word), vocabulary)
    
    # Jika masih tidak ada, kembalikan kata asli
    if not candidates_edit1:
        return misspelled_word
    
    # Evaluasi berdasarkan konteks
    for candidate in candidates_edit1:
        # Skor unigram (frekuensi kata)
        unigram_score = word_counts[candidate]
        
        # Skor bigram dengan kata sebelumnya
        prev_bigram_score = bigram_counts.get((prev_word, candidate), 0) if prev_word else 0
        
        # Skor bigram dengan kata berikutnya
        next_bigram_score = bigram_counts.get((candidate, next_word), 0) if next_word else 0
        
        # Skor total dengan pembobotan
        total_score = unigram_score + 2 * prev_bigram_score + 2 * next_bigram_score
        
        candidates.append((candidate, total_score))
    
    # Pilih kata dengan skor tertinggi
    if candidates:
        best_candidate = max(candidates, key=lambda x: x[1])[0]
        return best_candidate
    
    return misspelled_word

def correct_text_with_context(text, word_counts, vocabulary, bigram_counts):
    """
    Memperbaiki ejaan dalam teks lengkap dengan mempertimbangkan konteks.
    
    Args:
        text (str): Teks yang akan diperbaiki
        word_counts (Counter): Frekuensi kata dalam korpus
        vocabulary (set): Kumpulan kata yang valid
        bigram_counts (Counter): Frekuensi bigram
        
    Returns:
        str: Teks yang telah diperbaiki
    """
    words = re.findall(r'\w+', text.lower())
    corrected_words = []
    
    for i, word in enumerate(words):
        prev_word = words[i-1] if i > 0 else None
        next_word = words[i+1] if i < len(words) - 1 else None
        
        corrected = predict_with_context(word, prev_word, next_word, 
                                         word_counts, vocabulary, bigram_counts)
        corrected_words.append(corrected)
    
    # Rekonstruksi teks
    result = text
    for original, corrected in zip(words, corrected_words):
        if original != corrected:
            # Ganti kata salah dengan koreksi, perhatikan perbedaan case
            result = re.sub(r'\b' + re.escape(original) + r'\b', corrected, result, flags=re.IGNORECASE)
    
    return result

# ==================== ANALISIS KESALAHAN ====================
def analyze_common_errors(test_data, corrector):
    """
    Menganalisis kesalahan umum dalam sistem koreksi ejaan.
    
    Args:
        test_data (list): Pasangan (kata_salah, kata_benar)
        corrector (function): Fungsi koreksi ejaan
        
    Returns:
        dict: Analisis kesalahan
    """
    error_types = {
        'substitution': 0,
        'insertion': 0,
        'deletion': 0,
        'transposition': 0,
        'multiple': 0,
        'unknown': 0
    }
    
    error_examples = {key: [] for key in error_types}
    
    for misspelled, correct in test_data:
        prediction = corrector(misspelled.lower())
        
        if prediction != correct.lower():
            # Analisis jenis kesalahan
            error_type = classify_error(misspelled, correct)
            error_types[error_type] += 1
            
            # Simpan contoh kesalahan (maksimal 5 per tipe)
            if len(error_examples[error_type]) < 5:
                error_examples[error_type].append({
                    'misspelled': misspelled,
                    'correct': correct,
                    'prediction': prediction
                })
    
    return {
        'error_counts': error_types,
        'error_examples': error_examples,
        'total_errors': sum(error_types.values())
    }

def classify_error(misspelled, correct):
    """
    Mengklasifikasi jenis kesalahan ejaan.
    
    Args:
        misspelled (str): Kata yang salah eja
        correct (str): Kata yang benar
        
    Returns:
        str: Jenis kesalahan
    """
    # Konversi ke lowercase
    misspelled = misspelled.lower()
    correct = correct.lower()
    
    # Panjang sama dengan satu karakter berbeda -> substitusi
    if len(misspelled) == len(correct):
        diff_count = sum(a != b for a, b in zip(misspelled, correct))
        if diff_count == 1:
            return 'substitution'
        elif diff_count == 2:
            # Periksa apakah transposisi (penukaran karakter yang berdekatan)
            for i in range(len(misspelled) - 1):
                if (misspelled[i] == correct[i+1] and misspelled[i+1] == correct[i] and
                    misspelled[:i] == correct[:i] and misspelled[i+2:] == correct[i+2:]):
                    return 'transposition'
    
    # Panjang berbeda satu dengan sebagian besar karakter sama -> insertion atau deletion
    elif len(misspelled) == len(correct) + 1:
        # Misspelled lebih panjang -> insertion
        for i in range(len(correct) + 1):
            if (misspelled[:i] == correct[:i] and 
                misspelled[i+1:] == correct[i:]):
                return 'insertion'
    
    elif len(misspelled) + 1 == len(correct):
        # Misspelled lebih pendek -> deletion
        for i in range(len(misspelled) + 1):
            if (correct[:i] == misspelled[:i] and 
                correct[i+1:] == misspelled[i:]):
                return 'deletion'
    
    # Jika ada lebih dari satu operasi edit
    if levenshtein_distance(misspelled, correct) > 2:
        return 'multiple'
    
    return 'unknown'

def visualize_levenshtein_matrix(word1, word2):
    """
    Menampilkan matriks jarak Levenshtein untuk visualisasi.
    
    Args:
        word1 (str): Kata pertama
        word2 (str): Kata kedua
    """
    matrix = levenshtein_distance_matrix(word1, word2)
    
    # Tampilkan header
    print("    " + " ".join(c for c in "  " + word2))
    
    # Tampilkan matriks dengan indeks baris
    for i, row in enumerate(matrix):
        if i == 0:
            print("  " + " ".join(str(cell) for cell in row))
        else:
            print(word1[i-1] + " " + " ".join(str(cell) for cell in row))
    
    print(f"\nJarak Levenshtein: {matrix[-1, -1]}")

# ==================== PEMROSESAN DATASET PENGUJIAN ====================
def load_test_data(file_path):
    """
    Loads test data from a file with format: correct_word: misspelled_word1 misspelled_word2 ...
    
    Args:
        file_path (str): Path to test file
        
    Returns:
        list: Pairs of (misspelled_word, correct_word)
    """
    test_pairs = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                line = line.strip()
                if ':' in line:
                    correct, misspelled_str = line.split(':', 1)
                    correct = correct.strip()
                    misspelled_words = misspelled_str.strip().split()
                    
                    for misspelled in misspelled_words:
                        test_pairs.append((misspelled.strip(), correct))
    except UnicodeDecodeError:
        # Try another encoding if utf-8 fails
        with open(file_path, 'r', encoding='latin-1') as file:
            for line in file:
                line = line.strip()
                if ':' in line:
                    correct, misspelled_str = line.split(':', 1)
                    correct = correct.strip()
                    misspelled_words = misspelled_str.strip().split()
                    
                    for misspelled in misspelled_words:
                        test_pairs.append((misspelled.strip(), correct))
    
    return test_pairs

def evaluate_correction_accuracy(corrector, test_data):
    """
    Mengevaluasi akurasi sistem koreksi ejaan.
    
    Args:
        corrector (function): Fungsi koreksi ejaan
        test_data (list): Pasangan (kata_salah, kata_benar)
        
    Returns:
        dict: Metrik evaluasi (akurasi, jumlah_benar, jumlah_total)
    """
    correct_count = 0
    total_count = len(test_data)
    
    for misspelled, correct in test_data:
        prediction = corrector(misspelled.lower())
        if prediction == correct.lower():
            correct_count += 1
    
    accuracy = correct_count / total_count if total_count > 0 else 0
    
    return {
        'accuracy': accuracy,
        'correct_count': correct_count,
        'total_count': total_count
    }

def evaluate_with_execution_time(func, *args, **kwargs):
    """
    Mengukur waktu eksekusi fungsi.
    
    Args:
        func (function): Fungsi yang akan diukur
        *args, **kwargs: Argumen untuk fungsi
        
    Returns:
        tuple: (hasil_fungsi, waktu_eksekusi_detik)
    """
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    
    return result, end_time - start_time

# ==================== FUNGSI UTAMA ====================
def main():
    """
    Fungsi utama untuk menjalankan dan menguji sistem koreksi ejaan.
    """

    # Inisialisasi output capture untuk menyimpan hasil evaluasi
    output_capture = OutputCapture()
    output_capture.start_capture()

    print("Memulai sistem koreksi ejaan...")
    
    # Memuat dan memproses korpus
    print("Memuat korpus...")
    corpus_path = 'data/big.txt'
    word_counts, raw_corpus = process_corpus(corpus_path)
    vocabulary = build_vocabulary(word_counts, min_frequency=1)
    print(f"Korpus dimuat: {len(word_counts)} kata unik, {sum(word_counts.values())} total kata")
    
    # Memuat bigram
    print("Menghitung frekuensi bigram...")
    bigram_counts = compute_ngram_frequencies(raw_corpus, n=2)
    print(f"Jumlah bigram unik: {len(bigram_counts)}")
    
    # Persiapkan fungsi koreksi
    def correct_misspelled_simple(word):
        return correct_word_levenshtein(word.lower(), word_counts, vocabulary)
    
    def correct_misspelled_with_context(word, prev=None, next=None):
        return predict_with_context(word.lower(), prev, next, word_counts, vocabulary, bigram_counts)
    
    # Memuat dan evaluasi data pengujian
    test_files = [
        'data/spell-testset1.txt',
        'data/spell-testset2.txt'
    ]

    print("\n========= EVALUASI DENGAN ALGORITMA LEVENSHTEIN DASAR =========")
    for test_file in test_files:
        try:
            file_name = os.path.basename(test_file)
            print(f"\nEvaluasi dengan file {file_name}...")
            test_data = load_test_data(test_file)
            
            if not test_data:
                print(f"  Tidak ada data uji yang valid dalam file {file_name}")
                continue
                
            print(f"  Jumlah pasangan kata uji: {len(test_data)}")
            
            # Evaluasi sistem
            eval_result, execution_time = evaluate_with_execution_time(
                evaluate_correction_accuracy, correct_misspelled_simple, test_data
            )
            
            print(f"  Akurasi: {eval_result['accuracy']:.4f}")
            print(f"  Jumlah benar: {eval_result['correct_count']} dari {eval_result['total_count']}")
            print(f"  Waktu eksekusi: {execution_time:.2f} detik")
            
            # Tampilkan beberapa contoh prediksi
            print("\n  Contoh prediksi:")
            for i, (misspelled, correct) in enumerate(test_data[:5]):
                prediction = correct_misspelled_simple(misspelled.lower())
                is_correct = "✓" if prediction == correct.lower() else "✗"
                print(f"    {i+1}. {misspelled} -> {prediction} (seharusnya {correct}) {is_correct}")
            
            # Analisis jenis kesalahan
            print("\n  Analisis jenis kesalahan:")
            error_analysis = analyze_common_errors(test_data, correct_misspelled_simple)
            total_errors = error_analysis['total_errors']
            
            # Tampilkan distribusi kesalahan
            for error_type, count in error_analysis['error_counts'].items():
                percentage = (count / total_errors * 100) if total_errors > 0 else 0
                print(f"    {error_type}: {count} ({percentage:.1f}%)")
            
            # Tampilkan contoh kesalahan
            print("\n  Contoh kesalahan per tipe:")
            for error_type, examples in error_analysis['error_examples'].items():
                if examples:
                    print(f"    {error_type}:")
                    for i, example in enumerate(examples[:3]):  # Tampilkan maksimal 3 contoh
                        print(f"      {i+1}. {example['misspelled']} -> {example['prediction']} (seharusnya {example['correct']})")
            
        except Exception as e:
            print(f"  Error saat mengevaluasi file {file_name}: {str(e)}")
    
    print("\n========= EVALUASI DENGAN KOREKSI BERBASIS KONTEKS =========")
    
    # Hanya gunakan file test yang lebih kecil untuk evaluasi konteks (menghemat waktu)
    for test_file in ['data/spell-testset1.txt', 'data/spell-testset2.txt']:
        try:
            file_name = os.path.basename(test_file)
            print(f"\nEvaluasi konteks dengan file {file_name}...")
            test_data = load_test_data(test_file)
            
            if not test_data:
                print(f"  Tidak ada data uji yang valid dalam file {file_name}")
                continue
                
            # Untuk evaluasi konteks, kita perlu membuat conteks dummy untuk setiap kata
            # Dalam kasus nyata, konteks akan diambil dari kalimat
            contextualized_test_data = []
            
            for misspelled, correct in test_data:
                # Gunakan "the" sebagai konteks dummy
                contextualized_test_data.append((misspelled, "the", "the", correct))
            
            # Fungsi untuk mengevaluasi koreksi berbasis konteks
            def evaluate_context_correction(contextualized_data):
                correct_count = 0
                total_count = len(contextualized_data)
                
                for misspelled, prev, next, correct in contextualized_data:
                    prediction = correct_misspelled_with_context(misspelled.lower(), prev, next)
                    if prediction == correct.lower():
                        correct_count += 1
                
                accuracy = correct_count / total_count if total_count > 0 else 0
                
                return {
                    'accuracy': accuracy,
                    'correct_count': correct_count,
                    'total_count': total_count
                }
            
            # Evaluasi sistem berbasis konteks
            eval_result, execution_time = evaluate_with_execution_time(
                evaluate_context_correction, contextualized_test_data
            )
            
            print(f"  Akurasi dengan konteks: {eval_result['accuracy']:.4f}")
            print(f"  Jumlah benar: {eval_result['correct_count']} dari {eval_result['total_count']}")
            print(f"  Waktu eksekusi: {execution_time:.2f} detik")
            
            # Tampilkan beberapa contoh prediksi
            print("\n  Contoh prediksi dengan konteks:")
            for i, (misspelled, prev, next, correct) in enumerate(contextualized_test_data[:5]):
                prediction = correct_misspelled_with_context(misspelled.lower(), prev, next)
                is_correct = "✓" if prediction == correct.lower() else "✗"
                print(f"    {i+1}. {misspelled} -> {prediction} (seharusnya {correct}) {is_correct}")
            
        except Exception as e:
            print(f"  Error saat mengevaluasi file {file_name} dengan konteks: {str(e)}")
    
    # Berhenti capture dan simpan hasil evaluasi ke file
    output_capture.stop_capture()
    
    # Buat nama file dengan timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"result/hasil_evaluasi_{timestamp}.txt"
    
    # Simpan hasil ke file
    output_capture.save_to_file(output_filename)

    # Demo interaktif
    print("\n========= DEMO INTERAKTIF KOREKSI EJAAN =========")
    print("Ketik 'exit' untuk keluar, 'matrix' untuk visualisasi, atau 'context' untuk koreksi konteks.")
    current_mode = "simple"
    
    while True:
        if current_mode == "simple":
            user_input = input("\nMasukkan kata yang ingin dikoreksi (simple mode): ")
        elif current_mode == "context":
            user_input = input("\nMasukkan kata yang ingin dikoreksi (context mode): ")
        elif current_mode == "matrix":
            user_input = input("\nMasukkan dua kata untuk visualisasi matriks (pisahkan dengan spasi): ")
        
        if user_input.lower() == 'exit':
            break
        elif user_input.lower() == 'simple':
            current_mode = "simple"
            print("Beralih ke mode koreksi sederhana.")
            continue
        elif user_input.lower() == 'context':
            current_mode = "context"
            print("Beralih ke mode koreksi konteks. Masukkan: [kata_salah] [kata_sebelum] [kata_sesudah]")
            continue
        elif user_input.lower() == 'matrix':
            current_mode = "matrix"
            print("Beralih ke mode visualisasi matriks. Masukkan: [kata1] [kata2]")
            continue
        
        if current_mode == "simple":
            corrected = correct_misspelled_simple(user_input)
            distance = levenshtein_distance(user_input, corrected)
            print(f"Koreksi: {corrected} (jarak Levenshtein: {distance})")
            
        elif current_mode == "context":
            parts = user_input.split()
            if len(parts) >= 3:
                misspelled, prev, next = parts[0], parts[1], parts[2]
                corrected = correct_misspelled_with_context(misspelled, prev, next)
                print(f"Koreksi dengan konteks: {corrected}")
            else:
                print("Format salah. Gunakan: [kata_salah] [kata_sebelum] [kata_sesudah]")
                
        elif current_mode == "matrix":
            parts = user_input.split()
            if len(parts) == 2:
                word1, word2 = parts
                visualize_levenshtein_matrix(word1, word2)
            else:
                print("Format salah. Gunakan: [kata1] [kata2]")

if __name__ == "__main__":
    main()