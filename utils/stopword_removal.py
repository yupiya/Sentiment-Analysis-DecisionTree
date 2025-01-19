from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory, StopWordRemover, ArrayDictionary

# Stopwords tambahan
more_stop_words = [
    " toh ", " lah ", " yang ", " loh ", " dan ", " amp ", " sih ", " nya ", " pun ", " lagi ",
    " kok ", " tapi ", " saja ", " nah ", " nih ", " deh ", " oh ", " iya ", " eh ", " buat ",
    " dari ", " atau ", " ini ", " itu ", " pada ", " tersebut ", " kalau ", " kan ", " dengan ",
    " untuk ", " ke ", " juga ", " sudah ", " belum ", " hampir ", " harus ", " boleh ", " sangat ",
    " biar ", " bagaimana ", " kepada ", " karena ", " maka ", " agar ", " demi ", " apakah ",
    " ya ", " sedang ", " tentu ", " bila ", " jadi ", " begitu ", " sekitar ", " banyak ",
    " mungkin ", " sebagai ", " yaitu ", " misalnya ", " hanya ", " hampir ", " mereka ", " siapa ",
    " saat ", " masih ", " meski ", " oleh ", " tiap ", " semua ", " sebagian ", " hingga ", " supaya ",
    " yakni ", " walau ", " bahwa ", " dalam ", " tersebut ", " antara ", " dahulu ", " tanpa "
]

# Mengambil stopwords dari Sastrawi dan menambahkan stopwords tambahan
stop_words = StopWordRemoverFactory().get_stop_words()
stop_words.extend(more_stop_words)

# Pastikan "tidak" tidak termasuk dalam stopwords
if 'tidak' in stop_words:
    stop_words.remove('tidak')

# Membuat kamus baru untuk stopwords
new_array = ArrayDictionary(stop_words)
stop_words_remover_new = StopWordRemover(new_array)

# Fungsi untuk menghapus stopwords
def stopword(str_text):
    return stop_words_remover_new.remove(str_text)
