# Normalisasi text dictionary
norm = {
    " tndr ": " tender ",
    " prusahaan ": " perusahaan ",
    " plihara ": " pelihara ",
    " pembacaan ": " pembacaan ",
    " wajibcerita ": " wajib cerita ",
    " melakukan ": " lakukan ",
    " kmdn ": " kemudian ",
    " bbrp ": " beberapa ",
    " jnis ": " jenis ",
    " mlik ": " milik ",
    " pertanyaan2 ": " pertanyaan ",
    " bnyk ": " banyak ",
    " peryataan ": " pernyataan ",
    " smga ": " semoga ",
    " pola2 ": " pola ",
    " mliki ": " miliki ",
    " yth ": " yang terhormat ",
    " mnrt ": " menurut ",
    " jwbn ": " jawaban ",
    " mlh ": " malah ",
    " smpe ": " sampai ",
    " klmpok ": " kelompok ",
    " sndri ": " sendiri ",
    " kbrdaan ": " keberadaan ",
    " pke ": " pakai ",
    " kluar ": " keluar ",
    " brngkt ": " berangkat ",
    " slalu ": " selalu ",
    " mndpat ": " mendapat ",
    " mnggunakan ": " menggunakan ",
    " kecerahan ": " kecantikan ",
    " bc ": " baca ",
    " klo ": " kalau ",
    " bkan ": " bahkan ",
    " mnjd ": " menjadi ",
    " bw ": " bawakan ",
    " jwban ": " jawaban ",
    " kdri ": " kediri ",
    " bwat ": " buat ",
    " mjdi ": " menjadi ",
    " trsbt ": " tersebut ",
    " trlihat ": " terlihat ",
    " skrng ": " sekarang ",
    " ndiri ": " sendiri ",
    " mndkung ": " mendukung ",
    " ps ": " pasar ",
    " bs ": " bisa ",
    " kl ": " kalau ",
    " msk ": " masuk ",
    " smkn ": " semakin ",
    " mereka ": " mereka ",
    " sbb ": " sebab ",
    " mngn": " mengen ",
    " yg ": " yang ",
    " msh ": " masih ",
    " dlm ": " dalam ",
    " jwb": "jawab",
    " ini ": " ini ",
    " sbab": "sebab",
    " dpt ": " dapat ",
    " mbrikan ": " memberi ",
    " slmt": " selamat ",
    " smua ": " semua ",
    " tw ": " tahu ",
    " tnpa ": " tanpa ",
    " sdngkn ": " sedangkan ",
    " jd ": " jadi ",
    " hrs ": " harus ",
    " hny ": " hanya ",
    " dgn ": " dengan ",
    " mshnya ": " masih ",
    " tp ": " tapi ",
    " kmu": " kamu ",
    " dgn ": " dengan ",
    " ini ": " ini ",
    " tdk ": " tidak ",
    " trhdp ": " terhadap ",
    " msrakan ": " merasakan ",
    " batasan ": " batasan ",
    " mksh ": " terima kasih ",
    " dn ": " dan ",
    " mngadkan": "mengadakan",
    " krn ": " karena ",
    " mlh ": " malah ",
    " brrkaitan ": " berkaitan ",
    " trsbt ": " tersebut ",
    " trsbt ": " tersebut ",
    " mknnya ": " makanan ",
    " hrusnya": "harus",
    " pd": " pada ",
    "hri": "hari",
    "bnk":" banyak",
    " pnns ": " panas ",
    " bangun ": " bangunan ",
    "prkmhn": " perkembangan ",
    "membgi": " membagi ",
    "negara ": " nusantara ",
    "trdapat": " terdapat ",
    "pembbgian": " pembagian ",
    "kegiatan": " kegiatan ",
    "menempuh": " tempuh ",
    " sjak ": " sejak ",
    " sw": " sendiri",
    " yt ": " yaitu",
    " yoi": " ya",
    "gue": " saya",
    "gw": " saya",
    "lu":" anda",
    "lo":" anda",
    "loe":" anda",
    "hrsnya":" harusnya",
    "menhub":" kementrian perhubungan",
    "kereeenn":" keren",
    "hut":" hari ulang tahun"
}

# Fungsi normalisasi
def normalisasi(str_text):
    for key in norm:
        # Memeriksa apakah kata kunci memiliki spasi di sekitarnya
        if key.strip() in str_text:
            # Mengganti kata kunci dengan hasil penggantian mempertahankan spasi
            str_text = str_text.replace(key.strip(), norm[key].strip())
    return str_text