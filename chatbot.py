import os
import csv
import re
import requests
from deep_translator import GoogleTranslator

_SYMPTOM_VOCAB = None


def _symptom_vocab():
    global _SYMPTOM_VOCAB
    if _SYMPTOM_VOCAB is None:
        path = os.path.join(os.path.dirname(__file__), "data", "symptoms_dataset.csv")
        codes = set()
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                for k in ("symptom1", "symptom2", "symptom3", "symptom4", "symptom5"):
                    v = (row.get(k) or "").strip()
                    if v:
                        codes.add(v)
        _SYMPTOM_VOCAB = sorted(codes, key=len, reverse=True)
    return _SYMPTOM_VOCAB


def _codes_from_free_text(text_lower):
    allowed = set(_symptom_vocab())
    found = set()
    for sym in _symptom_vocab():
        spaced = sym.replace("_", " ")
        if len(spaced) >= 4 and spaced in text_lower:
            found.add(sym)
            continue
        if "_" in sym and sym in text_lower:
            found.add(sym)
    tokens = re.findall(r"[a-z][a-z0-9_]*", text_lower.replace("-", "_"))
    for t in tokens:
        if t in allowed:
            found.add(t)
    return found

# ─── EMERGENCY KEYWORDS ──────────────────────────────────────────
EMERGENCY_KEYWORDS = [
    'emergency', 'dying', 'cant breathe', "can't breathe", 'unconscious',
    'not breathing', 'heart attack', 'stroke', 'seizure', 'fainted',
    'collapsed', 'severe bleeding', 'overdose', 'poisoning', 'accident',
    'chest pain severe', 'no pulse', 'not responding', 'help me',
    # Tamil
    'udhavi', 'avasaram', 'moochu varala', 'nerinju vali',
    # Hindi
    'bachao', 'madad karo', 'sans nahi', 'behosh', 'dil ka daura'
]

EMERGENCY_RESPONSE = """🚨 EMERGENCY ALERT 🚨

Please call immediately:
🚑 Ambulance: 108
🏥 Medical Emergency: 102
👮 Police: 100
📞 National Health Helpline: 1800-180-1104

⚠️ Do NOT wait — call 108 RIGHT NOW!

While waiting for ambulance:
- Keep the person calm and still
- Do NOT give food or water
- Loosen tight clothing
- If unconscious — lay them on their side
- Stay on phone with 108 operator

🏥 Go to nearest government hospital emergency immediately."""

# ─── SYMPTOM KEYWORDS (natural language → symptom) ──────────────
SYMPTOM_KEYWORDS = {
    'fever': ['fever', 'temperature', 'hot body', 'burning body', 'kaichal', 'bukhar',
              'jvaram', 'feeling hot', 'high temp', 'body heat', 'thapu'],
    'cough': ['cough', 'coughing', 'kasi', 'khansi', 'khasi', 'dry cough', 'wet cough',
              'cough a lot', 'keep coughing'],
    'headache': ['headache', 'head pain', 'head ache', 'talai vali', 'sir dard',
                 'head hurts', 'my head', 'throbbing head', 'head is paining'],
    'fatigue': ['tired', 'fatigue', 'weakness', 'exhausted', 'no energy', 'thayarvu',
                'thakaan', 'very weak', 'feeling weak', 'dull', 'lethargic', 'no strength'],
    'nausea': ['nausea', 'feel like vomiting', 'sick feeling', 'vanthi uravu', 'matli',
               'want to vomit', 'queasy', 'uneasy stomach'],
    'vomiting': ['vomiting', 'vomit', 'threw up', 'vanthi', 'ulti', 'puking', 'throwing up'],
    'diarrhea': ['diarrhea', 'loose stool', 'loose motion', 'paeyal', 'dast',
                 'watery stool', 'running stomach', 'stomach running'],
    'chest_pain': ['chest pain', 'chest ache', 'heart pain', 'maarbu vali',
                   'seene mein dard', 'chest hurts', 'pain in chest', 'chest pressure',
                   'tightness in chest'],
    'breathing_difficulty': ['breathing difficulty', 'cant breathe', "can't breathe",
                             'short of breath', 'moochu tirumbu', 'saans lene mein takleef',
                             'breathless', 'difficulty breathing', 'hard to breathe',
                             'shortness of breath', 'wheezing'],
    'joint_pain': ['joint pain', 'knee pain', 'body pain', 'moopu vali',
                   'jodo mein dard', 'bone pain', 'aching joints', 'joints hurt'],
    'rash': ['rash', 'skin rash', 'red spots', 'tolu padai', 'spots on skin',
             'skin irritation', 'red patches', 'bumps on skin'],
    'itching': ['itching', 'itch', 'scratching', 'arippu', 'khujli', 'skin itching',
                'itchy', 'my skin itches'],
    'stomach_pain': ['stomach pain', 'belly pain', 'abdominal pain', 'vayiru vali',
                     'pet dard', 'stomach hurts', 'pain in stomach', 'tummy pain',
                     'stomach ache', 'my stomach', 'gastric pain'],
    'dizziness': ['dizziness', 'dizzy', 'spinning', 'thalai suzhal', 'chakkar',
                  'head spinning', 'vertigo', 'feeling dizzy', 'giddy'],
    'swelling': ['swelling', 'swollen', 'puffiness', 'veekkam', 'sujan',
                 'bloated', 'puffed up', 'swelled'],
    'burning_urination': ['burning urination', 'pain while urinating', 'burning pee',
                          'siru neer ericherpal', 'peshab mein jalan', 'painful urination',
                          'burning when i pee', 'urine burning'],
    'loss_of_appetite': ['no appetite', 'not hungry', 'loss of appetite', 'pasi illai',
                         'bhook nahi', 'not feeling like eating', 'dont want to eat',
                         "can't eat", 'no hunger'],
    'sadness': ['sad', 'depressed', 'crying', 'hopeless', 'kavalai', 'udaas',
                'feeling low', 'feel sad', 'unhappy', 'miserable'],
    'anxiety': ['anxious', 'worried', 'nervous', 'panic', 'paye', 'chinta',
                'feeling anxious', 'panicking', 'scared', 'fear'],
    'yellow_skin': ['yellow skin', 'yellow eyes', 'jaundice', 'manjal tolu',
                    'peeli aankhen', 'skin turned yellow', 'eyes are yellow'],
    'high_fever': ['high fever', 'very hot', 'severe fever', 'adhiga kaichal',
                   'tej bukhar', '103 fever', '104 fever', '105 fever', 'very high temp'],
    'body_ache': ['body ache', 'body pain', 'muscle pain', 'aching', 'udal vali',
                  'badan dard', 'all body pain', 'full body pain', 'muscles hurt'],
    'sore_throat': ['sore throat', 'throat pain', 'throat ache', 'thondai vali',
                    'gale mein dard', 'throat hurts', 'pain in throat', 'swollen throat'],
    'runny_nose': ['runny nose', 'nose running', 'nasal discharge', 'mooku ootruhal',
                   'naak bahna', 'nose dripping', 'running nose'],
    'weight_loss': ['weight loss', 'losing weight', 'getting thin', 'edai kuraivu',
                    'wajan kam hona', 'lost weight', 'losing a lot of weight'],
    'frequent_urination': ['frequent urination', 'urinating often', 'peeing a lot',
                           'adikadi siru neer', 'baar baar peshab', 'going to bathroom often',
                           'too many times bathroom'],
    'excessive_thirst': ['very thirsty', 'excessive thirst', 'drinking too much water',
                         'adhiga thagam', 'zyada pyaas', 'always thirsty', 'keep drinking water'],
    'sleep_problems': ['cant sleep', 'insomnia', 'sleep issues', 'thookam illai',
                       'neend nahi aati', 'not sleeping', 'sleeping too much', 'sleep problem'],
    'red_eyes': ['red eyes', 'eye redness', 'pink eye', 'kannin sivappu',
                 'aankhein laal', 'eyes are red', 'eyes burning'],
    'cold': ['cold', 'chills', 'shivering', 'feeling cold', 'shaking with cold'],
    'sneezing': ['sneezing', 'sneezes', 'sneezing a lot', 'thummal'],
    'congestion': ['congestion', 'blocked nose', 'stuffy nose', 'nose blocked'],
    'loss_of_smell': ['lost smell', 'cant smell', 'no smell', 'loss of smell'],
    'loss_of_taste': ['lost taste', 'cant taste', 'no taste', 'food has no taste'],
    'muscle_pain': ['muscle pain', 'muscle ache', 'muscles hurt', 'sore muscles'],
    'back_pain': ['back pain', 'lower back pain', 'back hurts', 'spine pain'],
    'skin_inflammation': ['skin inflammation', 'inflamed skin', 'skin swollen', 'skin red'],
    'pale_skin': ['pale skin', 'pale face', 'skin looks pale', 'whitish skin'],
    'irregular_heartbeat': ['heart racing', 'palpitations', 'irregular heartbeat',
                            'heart beating fast', 'fast heartbeat'],
    'numbness': ['numbness', 'numb', 'tingling', 'pins and needles', 'no feeling'],
    'blurred_vision': ['blurred vision', 'cant see clearly', 'vision problem',
                       'eyes blurry', 'blurry eyesight'],
    'blood_in_cough': ['blood in cough', 'coughing blood', 'blood when coughing'],
    'night_sweats': ['night sweats', 'sweating at night', 'wake up sweating'],
}

FOLLOW_UP_QUESTIONS = {
    'fever': 'Do you also have body pain or chills along with fever?',
    'headache': 'Is the headache mild or very severe? Do you have nausea too?',
    'cough': 'Is the cough dry or with phlegm? Do you have breathing difficulty?',
    'stomach_pain': 'Do you have nausea, vomiting, or diarrhea along with stomach pain?',
    'rash': 'Do you have itching with the rash? Is there fever too?',
    'chest_pain': 'Is the chest pain severe? Do you have shortness of breath?',
    'fatigue': 'How many days have you been feeling tired? Do you have any other symptoms?',
    'dizziness': 'Are you also having headache or nausea along with dizziness?',
    'breathing_difficulty': 'Do you have wheezing or tightness in the chest?',
}

def is_emergency(text):
    text_lower = text.lower()
    return any(keyword in text_lower for keyword in EMERGENCY_KEYWORDS)

def detect_language(text):
    tamil_chars = set('அஆஇஈஉஊஏஐஒஓஔகசடணதநபமயரலவழளறன')
    hindi_chars = set('अआइईउऊएऐओऔकखगघचछजझटठडढणतथदधनपफबभमयरलवशषसह')
    if any(c in tamil_chars for c in text):
        return 'ta'
    if any(c in hindi_chars for c in text):
        return 'hi'
    return 'en'

def translate_to_english(text, source_lang):
    if source_lang == 'en':
        return text
    try:
        return GoogleTranslator(source=source_lang, target='en').translate(text)
    except:
        return text

def translate_from_english(text, target_lang):
    if target_lang == 'en':
        return text
    try:
        return GoogleTranslator(source='en', target=target_lang).translate(text)
    except:
        return text

def extract_symptoms_from_text(text):
    text_lower = text.lower()
    # Translate if not English
    lang = detect_language(text)
    if lang != "en":
        text_lower = translate_to_english(text, lang).lower()
    found = []
    for symptom, keywords in SYMPTOM_KEYWORDS.items():
        for kw in keywords:
            if kw in text_lower:
                found.append(symptom)
                break
    found.extend(_codes_from_free_text(text_lower))
    return list(set(found))

def get_one_follow_up(symptoms):
    for s in symptoms:
        if s in FOLLOW_UP_QUESTIONS:
            return FOLLOW_UP_QUESTIONS[s]
    return "Can you describe any other discomfort you are feeling?"

def check_symptom_duration(duration_days):
    if duration_days > 7:
        return "⚠️ WARNING: Symptoms for over a week — see a doctor immediately!"
    elif duration_days > 3:
        return "⚠️ Symptoms for more than 3 days — please visit a clinic soon."
    return ""