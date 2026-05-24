from flask import Flask, render_template, request, jsonify, session, send_file
import os
import uuid
from chatbot import extract_symptoms_from_text, check_symptom_duration, is_emergency, EMERGENCY_RESPONSE
from predictor import predict_disease
from diet_plan import get_diet_plan
from health_tips import get_daily_tip, get_disease_tips, get_preventive_tips
from user_profile import create_profile, get_profile, add_consultation
from report_generator import generate_report
from location_service import get_nearby_places_detail
from season_context import (
    parse_season_from_text,
    season_prompt_message,
    SEASON_LABELS,
    client_season_value,
)

app = Flask(__name__)
app.secret_key = 'healthchatbot2024secretkey'
app.config['MAX_CONTENT_LENGTH'] = 12 * 1024 * 1024  # 12 MB uploads


def new_session():
    session['session_id'] = str(uuid.uuid4())
    session['symptoms'] = []
    session['step'] = 'symptoms'
    session['predictions'] = []
    session['severity'] = {}
    session['profile'] = {'name': 'User', 'age': 25, 'gender': 'Unknown'}
    session['season'] = None


def _duration_from_message(user_message):
    duration_warning = ""
    words = user_message.lower().split()
    for i, word in enumerate(words):
        if word.isdigit() and i + 1 < len(words) and 'day' in words[i + 1]:
            duration_warning = check_symptom_duration(int(word))
            break
    return duration_warning


def _render_prediction_payload(symptoms, duration_warning, predictions, severity, season_key=None):
    top_disease = predictions[0]['disease'] if predictions else "General Illness"
    profile = session.get('profile', {})
    diet = get_diet_plan(top_disease, age=profile.get('age', 25))
    tips = get_disease_tips(top_disease)
    preventive = get_preventive_tips(top_disease)
    pred_text = "\n".join([
        f"  • {p['disease']}: {p['probability']}%"
        for p in predictions
    ])
    eat_text = "\n".join([f"  ✅ {f}" for f in diet['foods_to_eat'][:5]])
    avoid_text = "\n".join([f"  ❌ {f}" for f in diet['foods_to_avoid'][:4]])
    tips_text = "\n".join([f"  💡 {t}" for t in tips[:3]])
    season_line = ""
    if season_key and season_key != "unknown":
        season_line = (
            f"\n🌦️ Season: {SEASON_LABELS.get(season_key, season_key)} "
            f"— possibilities lightly adjusted for typical seasonal patterns.\n"
        )
    response = f"""🔍 Symptoms detected: {', '.join(symptoms)}
{season_line}{duration_warning}

━━━━━━━━━━━━━━━━━━━━━━
🩺 POSSIBLE CONDITIONS:
{pred_text}

{severity['emoji']} SEVERITY: {severity['label']}
👉 {severity['action']}
━━━━━━━━━━━━━━━━━━━━━━
🥗 DIET PLAN ({top_disease}):

Foods to eat:
{eat_text}

Foods to avoid:
{avoid_text}
━━━━━━━━━━━━━━━━━━━━━━
💡 HEALTH TIPS:
{tips_text}

🛡️ Prevention: {preventive}
━━━━━━━━━━━━━━━━━━━━━━
What would you like next?
1️⃣ Generate PDF Report
2️⃣ Find Nearby Hospitals
3️⃣ Add More Symptoms
4️⃣ New Consultation"""
    return {
        'response': response,
        'step': 'results',
        'predictions': predictions,
        'severity': severity,
    }


@app.route('/')
def index():
    if 'session_id' not in session:
        new_session()
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json or {}
    step = session.get('step', 'symptoms')
    user_message = (data.get('message') or '').strip()
    if not user_message and step == 'season' and data.get('season'):
        user_message = str(data.get('season')).strip()

    if user_message == '__init__':
        new_session()
        tip = get_daily_tip()
        return jsonify({
            'response': f"""👋 Welcome to AI Health Assistant!

💡 Health Tip: {tip}

Just tell me your symptoms directly — no registration needed!

You can type in English, Tamil, or Hindi, use the microphone, or upload a symptom photo (optional AI vision — set an API key on the server).

Use the **Season** menu above the text box (or type winter / monsoon / skip in chat) so answers match the time of year.

Photos: after `pip install -r requirements.txt`, image upload uses a **local caption model** (no API key). Optional: Ollama or cloud keys for higher-quality vision.

Examples:
- "I have fever and headache"
- "என் வயிறு வலிக்கிறது"
- "Mujhe bukhar aur khasi hai"

🚨 Emergency? Type EMERGENCY or call 108 now!

What symptoms are you having? 🤒""",
            'step': 'symptoms'
        })

    if not user_message:
        return jsonify({
            'response': 'Please describe your symptoms and/or choose a season from the menu, then press Send.',
            'step': step,
        })

    session_id = session.get('session_id', str(uuid.uuid4()))
    step = session.get('step', 'symptoms')
    symptoms = session.get('symptoms', [])

    # ── EMERGENCY CHECK ALWAYS FIRST ─────────────────────────────
    if is_emergency(user_message):
        return jsonify({
            'response': EMERGENCY_RESPONSE,
            'step': 'emergency',
            'emergency': True
        })

    # ── SEASON (answered after symptoms) ─────────────────────────
    if step == 'season':
        p = parse_season_from_text(user_message)
        if not p:
            p = client_season_value(data.get('season'))
        extra = extract_symptoms_from_text(user_message)
        if extra:
            symptoms = list(set(session.get('symptoms', []) + extra))
            session['symptoms'] = symptoms
        symptoms = session.get('symptoms', [])
        if not p:
            return jsonify({
                'response': (
                    "I didn't catch the season. Reply with **winter**, **summer**, **monsoon**, "
                    "**spring**, **autumn**, or **skip**.\n\n" + season_prompt_message()
                ),
                'step': 'season',
            })
        session['season'] = p
        pred_season = None if p == "unknown" else p
        predictions, severity = predict_disease(symptoms, season=pred_season)
        session['predictions'] = predictions
        session['severity'] = severity
        session['step'] = 'results'
        dw = _duration_from_message(user_message)
        return jsonify(_render_prediction_payload(symptoms, dw, predictions, severity, season_key=p))

    # ── LOCATION STEP ─────────────────────────────────────────────
    if step == 'location':
        detail = get_nearby_places_detail(user_message, use_coords=False)
        session['step'] = 'results'
        return jsonify({'response': detail['text'], 'step': 'results', 'map': detail})

    # ── RESULTS OPTIONS ───────────────────────────────────────────
    if step == 'results':
        msg = user_message.lower()
        if '1' in msg or 'pdf' in msg or 'report' in msg:
            return make_pdf(session_id)
        elif '2' in msg or 'hospital' in msg or 'nearby' in msg:
            session['step'] = 'location'
            return jsonify({
                'response': "📍 Type your city name to find nearby hospitals:\nExample: Chennai, Mumbai, Delhi, Bangalore",
                'step': 'location'
            })
        elif '3' in msg or 'more' in msg or 'add' in msg:
            session['step'] = 'symptoms'
            session['season'] = None
            return jsonify({
                'response': "Tell me your additional symptoms. After that we will ask the season again.",
                'step': 'symptoms',
            })
        elif '4' in msg or 'new' in msg:
            new_session()
            return jsonify({'response': "Starting fresh! 🔄\n\nWhat symptoms are you having?", 'step': 'symptoms'})

    # ── SYMPTOM DETECTION & PREDICTION ───────────────────────────
    duration_warning = _duration_from_message(user_message)
    new_symptoms = extract_symptoms_from_text(user_message)
    symptoms = list(set(symptoms + new_symptoms))
    session['symptoms'] = symptoms

    cs = client_season_value(data.get('season'))
    if cs is not None:
        session['season'] = cs
    elif new_symptoms:
        sp = parse_season_from_text(user_message)
        if sp:
            session['season'] = sp

    if len(symptoms) >= 1:
        if session.get('season') is None:
            session['step'] = 'season'
            return jsonify({
                'response': (
                    f"🔍 Symptoms detected: {', '.join(symptoms)}\n\n" + season_prompt_message()
                ),
                'step': 'season',
            })
        pred_season = None if session['season'] == 'unknown' else session['season']
        predictions, severity = predict_disease(symptoms, season=pred_season)
        session['predictions'] = predictions
        session['severity'] = severity
        session['step'] = 'results'
        return jsonify(
            _render_prediction_payload(
                symptoms, duration_warning, predictions, severity, season_key=session.get('season')
            )
        )

    else:
        return jsonify({
            'response': """I couldn't detect symptoms from that. Please describe how you feel.

Try saying:
- "I have fever and headache"
- "My stomach hurts and I feel nauseous"
- "I have cough and body pain since 3 days"
- "என் தலை வலிக்கிறது" (Tamil)
- "Mujhe bukhar hai" (Hindi)

What exactly are you feeling? 🤒""",
            'step': 'symptoms'
        })

def make_pdf(session_id):
    try:
        profile = session.get('profile', {'name': 'User', 'age': 25, 'gender': 'Unknown'})
        predictions = session.get('predictions', [])
        severity = session.get('severity', {})
        symptoms = session.get('symptoms', [])
        top_disease = predictions[0]['disease'] if predictions else "General"
        diet = get_diet_plan(top_disease)
        filepath = generate_report(profile, symptoms, predictions, severity, diet, session_id)
        session['report_path'] = filepath
        return jsonify({
            'response': "✅ Your PDF health report is ready!\n\nClick the 📄 Download Report button below.",
            'step': 'results',
            'report_ready': True
        })
    except Exception as e:
        return jsonify({'response': f"❌ Report error: {str(e)}", 'step': 'results'})

@app.route('/download_report')
def download_report():
    report_path = session.get('report_path', '')
    if report_path and os.path.exists(report_path):
        return send_file(report_path, as_attachment=True,
                        download_name='health_report.pdf')
    return "Report not found. Please generate it first.", 404

@app.route('/get_location_hospitals', methods=['POST'])
def get_location_hospitals():
    data = request.json or {}
    lat = data.get('lat')
    lng = data.get('lng')
    detail = get_nearby_places_detail(f"{lat},{lng}", use_coords=True)
    return jsonify({'response': detail['text'], 'map': detail})


@app.route('/analyze_image', methods=['POST'])
def analyze_image():
    from image_analyzer import analyze_image_for_symptoms

    if 'session_id' not in session:
        new_session()
    upload = request.files.get('image')
    if not upload or not upload.filename:
        return jsonify({'response': 'No image file received.', 'step': 'symptoms'}), 400
    raw = upload.read()
    result = analyze_image_for_symptoms(raw, upload.mimetype or 'image/jpeg')
    if not result.get('ok'):
        msg = result.get('hint') or 'Could not analyze the image.'
        return jsonify({
            'response': f"📷 {msg}",
            'step': 'symptoms',
            'image_error': result.get('error'),
        })

    vision_syms = result.get('symptoms') or []
    notes = result.get('notes') or ''
    provider = result.get('provider') or 'vision'
    prior = session.get('symptoms', [])
    symptoms = list(dict.fromkeys(prior + vision_syms))
    session['symptoms'] = symptoms

    cs = client_season_value(request.form.get('season'))
    if cs is not None:
        session['season'] = cs

    if not symptoms:
        return jsonify({
            'response': (
                f"📷 Image analyzed ({provider}), but no symptom codes could be matched to our checklist.\n"
                f"Notes: {notes}\n\n"
                "Please describe what you feel (fever, rash, pain…) or try a clearer photo."
            ),
            'step': 'symptoms',
            'vision_notes': notes,
        })

    header = f"📷 From your image ({provider}): {', '.join(vision_syms) or 'general clues'}\n"
    if notes:
        header += f"📝 Notes: {notes}\n"

    if session.get('season') is None:
        session['step'] = 'season'
        return jsonify({
            'response': (
                header
                + f"🔍 Symptoms from image: {', '.join(symptoms)}\n\n"
                + season_prompt_message()
            ),
            'step': 'season',
        })

    pred_season = None if session.get('season') == 'unknown' else session.get('season')
    predictions, severity = predict_disease(symptoms, season=pred_season)
    session['predictions'] = predictions
    session['severity'] = severity
    session['step'] = 'results'
    duration_warning = ""
    payload = _render_prediction_payload(
        symptoms, duration_warning, predictions, severity, season_key=session.get('season')
    )
    payload['response'] = header + "\n" + payload['response']
    return jsonify(payload)

@app.route('/reset')
def reset():
    session.clear()
    return jsonify({'status': 'reset'})

if __name__ == '__main__':
    # threaded=True: long BLIP / vision work on one request won't block the whole server
    app.run(debug=True, host='127.0.0.1', port=5000, threaded=True)