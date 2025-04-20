from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import re
import string
from nltk.tokenize import word_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import nltk
nltk.download('punkt')
from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import requests
import json

app = Flask(__name__)
CORS(app)  # Supaya bisa diakses dari Flutter

# Muat data produk pakaian (pastikan file ada di folder yang sama)
data = pd.read_excel('data_bersih_final.xlsx')  # Ganti sesuai nama file

# Gabungkan semua fitur jadi satu kolom teks
def combine_features(row):
    return f"{row['nama pakaian']} {row['harga']} {row['rating']} {row['deskripsi']}"

data['combined_features'] = data.apply(combine_features, axis=1)

# Vektorisasi menggunakan TF-IDF
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(data['combined_features'])

# Simpan hasil rekomendasi ke Firebase
def simpan_ke_firebase(data):
    firebase_url = "https://kohood-6c455-default-rtdb.asia-southeast1.firebasedatabase.app/rekomendasi.json"
    response = requests.post(firebase_url, json=data)
    return response.status_code

@app.route('/rekomendasi', methods=['POST'])
def rekomendasi():
    try:
        query = request.json.get('query')
        query_vector = vectorizer.transform([query])
        cosine_sim = cosine_similarity(query_vector, tfidf_matrix).flatten()

        # Ambil 5 produk paling mirip
        top_indices = cosine_sim.argsort()[-5:][::-1]

        rekomendasi_produk = []

        for idx in top_indices:
            produk = {
                "contents href": data.iloc[idx]['contents href'],
                "nama pakaian": data.iloc[idx]['nama pakaian'],
                "harga": data.iloc[idx]['harga'],
                "rating": data.iloc[idx]['rating'],
            }
            rekomendasi_produk.append(produk)
            simpan_ke_firebase(produk)  # Simpan satu per satu ke Firebase

        return jsonify({"rekomendasi": rekomendasi_produk})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)






# client = openai.OpenAI(
#     api_key="sk-3360d2e9686148a8bc9f1f5aee621d99",
#     base_url="https://api.deepseek.com",
# )

# system_prompt = """
# The user will provide some exam text. Please parse the "question" and "answer" and output them in JSON format. 

# EXAMPLE INPUT: 
# Which is the highest mountain in the world? Mount Everest.

# EXAMPLE JSON OUTPUT:
# {
#     "question": "Which is the highest mountain in the world?",
#     "answer": "Mount Everest"
# }
# """

# user_prompt = "Which is the longest river in the world? The Nile River."

# messages = [{"role": "system", "content": system_prompt},
#             {"role": "user", "content": user_prompt}]

# response = client.chat.completions.create(
#     model="deepseek-chat",
#     messages=messages,
#     response_format={
#         'type': 'json_object'
#     }
# )

# print(json.loads(response.choices[0].message.content))