from flask import Flask, request, jsonify, render_template
import onnxruntime as ort
import numpy as np
import os
import sqlite3
from datetime import datetime
from tokenizer import SimpleTokenizer

app = Flask(__name__)

MODEL_PATH = os.path.join('model', 'smartflow_nlp.onnx')
VOCAB_PATH  = os.path.join('model', 'vocab.txt')
DB_PATH     = 'smartflow.db'

print("Loading model...")
session   = ort.InferenceSession(MODEL_PATH)
tokenizer = SimpleTokenizer(VOCAB_PATH)
print("Model loaded successfully!")

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS history (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            text      TEXT NOT NULL,
            label     TEXT NOT NULL,
            confidence REAL NOT NULL,
            timestamp TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

init_db()

DISTRACTION_KEYWORDS = [
    'tiktok', 'instagram', 'reels', 'netflix',
    'youtube', 'game', 'movie', 'drama', 'fun',
    'chill', 'masti', 'scroll', 'viral', 'funny',
    'meme', 'gossip', 'sale', 'discount', 'win',
    'free', 'prize', 'offer', 'click here', 'shop',
    'khana khane', 'khaane', 'chalte hain', 'chaltay hain',
    'chalein', 'chalay chalte', 'ghumne', 'ghumain',
    'bahar chalte', 'bahar chaltay', 'bahar chalein',
    'kuch khane', 'kuch khaane', 'kha lete', 'kha letay',
    'chai pite', 'chai pitay', 'chai peete', 'chai lete',
    'so jate', 'so jatay', 'so lete', 'so letay',
    'araam', 'neend', 'beth lete', 'beth letay',
    'yaar', 'dost', 'party', 'hang out', 'hangout',
    'shopping', 'market chalte', 'bazar', 'mall',
    'time pass', 'timepass', 'bore', 'bakwaas',
    'kal kar lenge', 'baad mein', 'abhi nahi',
    'ek minute', 'thoda aur', 'bas thoda',
    'cricket dekh', 'match dekh', 'match dekhte',
    'drama dekh', 'serial dekh', 'film dekh',
    'phone pe', 'mobile pe', 'scroll karte',
    'ghoomne', 'trip', 'picnic', 'enjoy','cash grant', 'grant', 'biometric', 'touch point',
    'you have received', 'you have won', 'congratulations',
    'prize money', 'lucky winner', 'selected',
    'claim your', 'withdraw', 'withdrawal',
    'tid:', 'transaction id', 'ref no',
    'punjab', 'authority', 'nadra',
    'visit nearest', 'nearest branch',
    'free registration', 'apply now',
    'government scheme', 'benazir',
    'ehsaas', 'bise', 'matric result',
    'send your cnic', 'cnic number',
    'whatsapp us', 'call now',
]

IMPORTANT_KEYWORDS = [
    # Work & meetings
    'meeting', 'call', 'zoom', 'interview', 'presentation',
    'deadline', 'submit', 'submission', 'urgent', 'emergency',
    'project', 'report', 'assignment', 'task', 'work',
    'office', 'boss', 'manager', 'client', 'email',

    # Health
    'doctor', 'hospital', 'medicine', 'appointment',
    'test', 'result', 'surgery', 'checkup', 'dawai',
    'dawa', 'ilaj', 'bemari', 'takleef',

    # Finance
    'invoice', 'payment', 'bill', 'salary', 'bank',
    'account', 'transfer', 'loan', 'tax', 'fee',
    'paisa', 'rupee', 'amount', 'due',

    # Education
    'exam', 'paper', 'class', 'lecture', 'study',
    'homework', 'result', 'marks', 'grade', 'college',
    'university', 'school', 'imtihan', 'padhai',

    # Alerts & notifications
    'reminder', 'alert', 'warning', 'notice', 'important',
    'please', 'kindly', 'asap', 'immediately', 'today',
    'tomorrow', 'tonight', 'kal', 'aaj', 'abhi',

    # Pakistani important context
    'zaruri', 'zaroori', 'emergency', 'mushkil',
    'madad', 'help', 'problem', 'issue', 'kaam',
    'interview', 'result aaya', 'admit card',
    'form submit', 'last date', 'fee jama'
]

def classify_text(text):
    text_lower = text.lower()

    # Pehle distraction check karo
    for keyword in DISTRACTION_KEYWORDS:
        if keyword in text_lower:
            return 'Distraction', 95.0

    # Phir important check karo
    for keyword in IMPORTANT_KEYWORDS:
        if keyword in text_lower:
            # Model se confirm karo
            encoded        = tokenizer.encode(text, max_length=64)
            input_ids      = np.array([encoded['input_ids']],      dtype=np.int64)
            attention_mask = np.array([encoded['attention_mask']], dtype=np.int64)
            outputs        = session.run(['logits'], {
                'input_ids':      input_ids,
                'attention_mask': attention_mask
            })
            logits     = outputs[0][0]
            prediction = int(np.argmax(logits))
            confidence = float(np.max(np.exp(logits) / np.sum(np.exp(logits))) * 100)
            label      = 'Important' if prediction == 1 else 'Distraction'
            return label, round(confidence, 2)

    # Na distraction na important keywords mile — default Distraction
    return 'Distraction', 75.0

def save_to_db(text, label, confidence):
    conn = sqlite3.connect(DB_PATH)
    c    = conn.cursor()
    c.execute(
        'INSERT INTO history (text, label, confidence, timestamp) VALUES (?, ?, ?, ?)',
        (text, label, confidence, datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    )
    conn.commit()
    conn.close()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data  = request.get_json()
    text  = data.get('text', '').strip()
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    label, confidence = classify_text(text)
    save_to_db(text, label, confidence)
    return jsonify({'label': label, 'confidence': confidence, 'text': text})

@app.route('/batch', methods=['POST'])
def batch():
    data  = request.get_json()
    texts = data.get('texts', [])
    if not texts:
        return jsonify({'error': 'No texts provided'}), 400
    results = []
    for text in texts:
        text = text.strip()
        if not text:
            continue
        label, confidence = classify_text(text)
        save_to_db(text, label, confidence)
        results.append({'text': text, 'label': label, 'confidence': confidence})
    return jsonify({'results': results, 'total': len(results)})

@app.route('/history', methods=['GET'])
def get_history():
    conn = sqlite3.connect(DB_PATH)
    c    = conn.cursor()
    c.execute('SELECT id, text, label, confidence, timestamp FROM history ORDER BY id DESC LIMIT 50')
    rows = c.fetchall()
    conn.close()
    history = [
        {'id': r[0], 'text': r[1], 'label': r[2], 'confidence': r[3], 'timestamp': r[4]}
        for r in rows
    ]
    return jsonify({'history': history})

@app.route('/stats', methods=['GET'])
def get_stats():
    conn = sqlite3.connect(DB_PATH)
    c    = conn.cursor()
    c.execute('SELECT COUNT(*) FROM history')
    total = c.fetchone()[0]
    c.execute("SELECT COUNT(*) FROM history WHERE label = 'Important'")
    important = c.fetchone()[0]
    c.execute("SELECT COUNT(*) FROM history WHERE label = 'Distraction'")
    distraction = c.fetchone()[0]
    c.execute('SELECT AVG(confidence) FROM history')
    avg_conf = c.fetchone()[0] or 0
    conn.close()
    return jsonify({
        'total':       total,
        'important':   important,
        'distraction': distraction,
        'avg_confidence': round(avg_conf, 2)
    })

@app.route('/clear_history', methods=['POST'])
def clear_history():
    conn = sqlite3.connect(DB_PATH)
    c    = conn.cursor()
    c.execute('DELETE FROM history')
    conn.commit()
    conn.close()
    return jsonify({'message': 'History cleared'})

if __name__ == '__main__':
    app.run(debug=True)