from __future__ import annotations

import os
import secrets
import threading
import math

from flask import Flask, jsonify, render_template, request

from learning_engine import (
    ACTION_DELTAS, QUESTIONS, bayesian_mastery, choose_q_action, create_session,
    q_state, train_classifier, update_q,
)

app = Flask(__name__)
app.config["JSON_SORT_KEYS"] = False
_sessions: dict[str, dict] = {}
_lock = threading.RLock()
_model, _metrics = train_classifier()


def error(message: str, status: int = 400):
    return jsonify({"error": message}), status


def session_for(data: dict):
    session_id = data.get("session_id")
    if not isinstance(session_id, str) or session_id not in _sessions:
        return None
    return _sessions[session_id]


@app.get("/")
def home():
    return render_template("index.html")


@app.get("/api/health")
def health():
    return jsonify({"status": "ok", "service": "Ақылды оқу көмекшісі"})


@app.post("/api/session")
def new_session():
    session_id = secrets.token_urlsafe(18)
    with _lock:
        _sessions[session_id] = create_session(seed=int.from_bytes(os.urandom(4), "big"))
    return jsonify({"session_id": session_id, "message": "Оқу сессиясы басталды."})


@app.post("/api/quiz/next")
def next_question():
    data = request.get_json(silent=True) or {}
    session = session_for(data)
    if session is None:
        return error("Оқу сессиясы табылмады. Бетті жаңартып, қайта бастаңыз.", 404)
    topic = data.get("topic")
    if topic not in QUESTIONS:
        return error("Тақырыпты тізімнен таңдаңыз.")
    with _lock:
        difficulty = session["difficulty"]
        candidates = QUESTIONS[topic][difficulty]
        position = session["topic_index"][topic] % len(candidates)
        session["topic_index"][topic] += 1
        source = candidates[position]
        qid = f"{topic}:{difficulty}:{position}"
        session["current"] = {"id": qid, "topic": topic, "difficulty": difficulty, **source}
        return jsonify({"id": qid, "topic": topic, "difficulty": difficulty,
                        "difficulty_label": ("Бастапқы", "Орташа", "Күрделі")[difficulty-1],
                        "prompt": source["prompt"], "choices": source["choices"],
                        "question_number": len(session["history"]) + 1})


@app.post("/api/quiz/answer")
def answer_question():
    data = request.get_json(silent=True) or {}
    session = session_for(data)
    if session is None:
        return error("Оқу сессиясы табылмады. Бетті жаңартып, қайта бастаңыз.", 404)
    with _lock:
        current = session["current"]
        if current is None or data.get("question_id") != current["id"]:
            return error("Бұл сұрақ ескірген. Келесі сұрақты қайта ашыңыз.", 409)
        choice = data.get("choice")
        if isinstance(choice, bool) or not isinstance(choice, int) or not 0 <= choice < len(current["choices"]):
            return error("Жауап нұсқасын таңдаңыз.")

        correct = choice == current["answer"]
        session["correct"] += int(correct)
        session["incorrect"] += int(not correct)
        session["streak"] = min(2, session["streak"] + 1) if correct else 0
        session["mastery"] = bayesian_mastery(.5, session["correct"], session["incorrect"])
        reward = 1.0 if correct else -1.0
        next_state = q_state(current["difficulty"], session["streak"])

        # Learn from the response to the question selected by the previous action.
        if session["pending"] is not None:
            old_state, old_action = session["pending"]
            update_q(session["q"], old_state, old_action, reward, next_state)

        action = choose_q_action(session["q"], next_state, session["rng"])
        next_difficulty = max(1, min(3, current["difficulty"] + ACTION_DELTAS[action]))
        session["pending"] = (next_state, action)
        session["difficulty"] = next_difficulty
        session["history"].append({"topic": current["topic"], "difficulty": current["difficulty"], "correct": correct, "reward": reward, "q_value": session["q"][(next_state[0], next_state[1], action)]})
        return jsonify({"correct": correct, "explanation": current["explanation"],
                        "mastery": round(session["mastery"], 4), "correct_count": session["correct"],
                        "incorrect_count": session["incorrect"], "reward": reward,
                        "next_difficulty": next_difficulty,
                        "next_difficulty_label": ("Бастапқы", "Орташа", "Күрделі")[next_difficulty-1],
                        "q_table": {"жеңілдету": round(session["q"][(next_state[0], next_state[1], 0)], 3),
                                    "сақтау": round(session["q"][(next_state[0], next_state[1], 1)], 3),
                                    "күрделендіру": round(session["q"][(next_state[0], next_state[1], 2)], 3)}})


@app.post("/api/bayes")
def bayes_endpoint():
    data = request.get_json(silent=True) or {}
    try:
        prior = float(data.get("prior", .5))
        correct, incorrect = data.get("correct"), data.get("incorrect")
        sensitivity = float(data.get("sensitivity", .8))
        guess = float(data.get("guess", .25))
        if isinstance(correct, bool) or not isinstance(correct, int) or isinstance(incorrect, bool) or not isinstance(incorrect, int):
            raise ValueError("Жауап сандары бүтін сан болуы керек.")
        posterior = bayesian_mastery(prior, correct, incorrect, sensitivity, guess)
    except (ValueError, TypeError, OverflowError) as exc:
        return error(str(exc))
    return jsonify({"posterior": round(posterior, 4), "percent": round(posterior * 100, 1),
                    "correct": correct, "incorrect": incorrect,
                    "interpretation": "Болжамдық оқу бағасы; ресми баға немесе диагноз емес."})


@app.post("/api/ml/predict")
def ml_predict():
    data = request.get_json(silent=True) or {}
    try:
        raw_practice = data.get("practice")
        if isinstance(raw_practice, bool) or not isinstance(raw_practice, int):
            raise ValueError
        practice = raw_practice
        accuracy = float(data.get("accuracy"))
        attendance = float(data.get("attendance"))
        if not 0 <= practice <= 20 or not math.isfinite(accuracy) or not math.isfinite(attendance) or not 0 <= accuracy <= 100 or not 0 <= attendance <= 100:
            raise ValueError("Жаттығу саны 0–20, ал пайыздар 0–100 аралығында болуы керек.")
    except (TypeError, ValueError):
        return error("Сандарды тексеріңіз: жаттығу саны 0–20, пайыздар 0–100 аралығында болуы керек.")
    predicted = str(_model.predict([[practice, accuracy, attendance]])[0])
    probabilities = _model.predict_proba([[practice, accuracy, attendance]])[0]
    confidence = float(max(probabilities))
    advice = {"Жоғары": "Күрделі есептермен біліміңізді тереңдетіңіз.",
              "Орташа": "Қысқа қайталау жасап, қосымша жаттығу орындаңыз.",
              "Қолдау қажет": "Негізгі ұғымдарды қайталап, жеңіл тапсырмадан бастаңыз."}[predicted]
    return jsonify({"prediction": predicted, "confidence": round(confidence, 3), "recommendation": advice,
                    "metrics": _metrics, "dataset": "Синтетикалық оқу үлгісі"})


@app.get("/api/ml/metrics")
def ml_metrics():
    return jsonify({"metrics": _metrics, "labels": ["Қолдау қажет", "Орташа", "Жоғары"],
                    "features": ["Жаттығу саны", "Тест дәлдігі", "Қатысу"], "dataset": "Синтетикалық оқу үлгісі"})


@app.get("/api/analytics")
def analytics():
    session_id = request.args.get("session_id", "")
    session = _sessions.get(session_id)
    if session is None:
        return error("Оқу сессиясы табылмады.", 404)
    total = session["correct"] + session["incorrect"]
    return jsonify({"total": total, "correct": session["correct"], "incorrect": session["incorrect"],
                    "accuracy": round(session["correct"] / total, 3) if total else 0,
                    "mastery": round(session["mastery"], 4), "history": session["history"][-12:]})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "5000")), debug=False)
