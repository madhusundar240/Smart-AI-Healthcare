import random
from datetime import datetime

DAILY_TIPS = [
    "💧 Drink at least 8 glasses of water today",
    "🚶 Walk for 30 minutes — even a slow walk improves health",
    "🥗 Eat one extra serving of green vegetables today",
    "😴 Sleep 7-8 hours tonight — your body heals during sleep",
    "🧘 Take 5 deep breaths right now — it reduces stress instantly",
    "🚭 Avoid smoking — it damages lungs, heart, and skin",
    "🌞 Get 15 minutes of sunlight daily for Vitamin D",
    "🧼 Wash hands properly before every meal",
    "🍎 Eat fruits instead of sweets for natural energy",
    "📵 Take a break from screens every 20 minutes",
    "🧂 Reduce salt in your food — it prevents high BP",
    "🫀 Check your blood pressure at least once a month",
    "🦷 Brush teeth twice daily — oral health affects heart health",
    "😊 Laugh more — it boosts immunity and reduces stress",
    "🏃 Exercise is free medicine — do it daily",
]

DISEASE_TIPS = {
    "Diabetes": [
        "Check your blood sugar every morning",
        "Never skip meals — hypoglycemia is dangerous",
        "Wear shoes always — even indoors — to protect feet",
        "Check your feet daily for any cuts or wounds",
    ],
    "Hypertension": [
        "Check BP every morning before eating",
        "Avoid anger and stress — they spike BP immediately",
        "Reduce salt to less than 5g per day",
        "Take medicines at the same time every day",
    ],
    "Asthma": [
        "Keep your inhaler with you at all times",
        "Avoid dust, smoke, and strong perfumes",
        "Do breathing exercises every morning",
        "Know your asthma triggers and avoid them",
    ],
    "Heart Disease": [
        "Never ignore chest pain — go to hospital immediately",
        "Take all medicines exactly as prescribed",
        "Avoid lifting heavy weights",
        "Sleep on your left side to reduce heart strain",
    ],
    "Dengue": [
        "Check platelet count daily during dengue",
        "Drink papaya leaf juice to boost platelets",
        "Watch for bleeding from gums or nose — go to hospital immediately",
        "Avoid aspirin or ibuprofen — use only paracetamol",
    ],
}

def get_daily_tip():
    day_of_year = datetime.now().timetuple().tm_yday
    tip = DAILY_TIPS[day_of_year % len(DAILY_TIPS)]
    return tip

def get_disease_tips(disease):
    tips = DISEASE_TIPS.get(disease, [])
    general = random.sample(DAILY_TIPS, min(3, len(DAILY_TIPS)))
    return tips + general

def get_preventive_tips(disease):
    preventive = {
        "Malaria": "Use mosquito nets while sleeping. Apply mosquito repellent. Remove stagnant water near your home.",
        "Dengue": "Use mosquito repellent. Wear full sleeves. Remove stagnant water from flower pots and coolers.",
        "COVID-19": "Wash hands frequently. Wear mask in crowded places. Maintain distance from sick people.",
        "Typhoid": "Drink only boiled or filtered water. Eat freshly cooked food. Wash hands before eating.",
        "Cholera": "Drink only safe water. Wash hands with soap. Avoid street food during outbreaks.",
        "Tuberculosis": "Ventilate your home. Cover mouth while coughing. Complete the full course of treatment.",
    }
    return preventive.get(disease, "Maintain good hygiene, eat balanced diet, exercise regularly, and visit doctor for regular checkups.")