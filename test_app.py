"""Offline API-dispatch, algorithm, static asset, and deployment configuration tests."""
import json
import os
import unittest
from pathlib import Path
from unittest.mock import patch

import local_server
from learning_engine import (
    _macro_metrics, bayesian_mastery, create_session, make_training_data,
    q_state, train_classifier, update_q,
)


class LocalResponse:
    def __init__(self, status, payload):
        self.status_code, self.payload = status, payload

    def get_json(self):
        return self.payload


class WebsiteTests(unittest.TestCase):
    def setUp(self):
        status, payload = local_server.dispatch("POST", "/api/session", {})
        self.assertEqual(status, 200)
        self.session_id = payload["session_id"]

    def call(self, method, path, payload=None):
        status, body = local_server.dispatch(method, path, payload)
        return LocalResponse(status, body)

    def post(self, path, payload):
        return self.call("POST", path, payload)

    def test_home_assets_and_offline_html(self):
        root = Path(__file__).parent
        html = (root / "templates" / "index.html").read_text(encoding="utf-8")
        css = (root / "static" / "css" / "style.css").read_text(encoding="utf-8")
        js = (root / "static" / "js" / "app.js").read_text(encoding="utf-8")
        self.assertIn('<html lang="kk">', html)
        self.assertIn("Басты бет", html)
        self.assertNotIn("fonts.googleapis.com", html)
        self.assertIn("@media", css)
        self.assertIn("/api/quiz/answer", js)

    def test_health_and_metrics_routes(self):
        status, health = local_server.dispatch("GET", "/api/health")
        self.assertEqual(status, 200)
        self.assertEqual(health["status"], "ok")
        status, result = local_server.dispatch("GET", "/api/ml/metrics")
        self.assertEqual(status, 200)
        self.assertGreaterEqual(result["metrics"]["test_size"], 58)
        self.assertLessEqual(result["metrics"]["test_size"], 61)
        self.assertTrue(0 <= result["metrics"]["f1"] <= 1)

    def test_session_quiz_q_learning_and_analytics_flow(self):
        status, q = local_server.dispatch("POST", "/api/quiz/next", {"session_id": self.session_id, "topic": "Q-learning"})
        self.assertEqual(status, 200)
        self.assertIn("prompt", q)
        status, result = local_server.dispatch("POST", "/api/quiz/answer", {"session_id": self.session_id, "question_id": q["id"], "choice": 0})
        self.assertEqual(status, 200)
        self.assertTrue(result["correct"])
        self.assertIn("күрделендіру", result["q_table"])
        status, analytics = local_server.dispatch("GET", "/api/analytics?session_id=" + self.session_id)
        self.assertEqual(status, 200)
        self.assertEqual((analytics["total"], analytics["correct"]), (1, 1))

    def test_quiz_validates_topic_session_choice_and_stale_question(self):
        self.assertEqual(self.post("/api/quiz/next", {"session_id": self.session_id, "topic": []}).status_code, 400)
        self.assertEqual(self.post("/api/quiz/next", {"session_id": "missing", "topic": "Q-learning"}).status_code, 404)
        q = self.post("/api/quiz/next", {"session_id": self.session_id, "topic": "Q-learning"}).get_json()
        self.assertEqual(self.post("/api/quiz/answer", {"session_id": self.session_id, "question_id": q["id"], "choice": 99}).status_code, 400)
        self.assertEqual(self.post("/api/quiz/answer", {"session_id": self.session_id, "question_id": "stale", "choice": 0}).status_code, 409)

    def test_bayesian_endpoint_and_validation(self):
        result = self.post("/api/bayes", {"prior": .5, "correct": 4, "incorrect": 1})
        self.assertEqual(result.status_code, 200)
        self.assertGreater(result.get_json()["posterior"], .5)
        self.assertEqual(self.post("/api/bayes", {"prior": .5, "correct": 0, "incorrect": 0}).status_code, 400)
        self.assertEqual(self.post("/api/bayes", {"prior": 1.4, "correct": 1, "incorrect": 0}).status_code, 400)
        self.assertGreater(bayesian_mastery(.5, 1, 0), .5)
        self.assertLess(bayesian_mastery(.5, 0, 1), .5)

    def test_classifier_is_local_deterministic_and_returns_predictions(self):
        self.assertEqual(make_training_data(), make_training_data())
        model, metrics = train_classifier()
        self.assertEqual(metrics["train_size"] + metrics["test_size"], 240)
        self.assertGreaterEqual(metrics["test_size"], 58)
        self.assertLessEqual(metrics["test_size"], 61)
        row = [[7, 80, 90]]
        self.assertEqual(len(model.predict(row)), 1)
        self.assertAlmostEqual(sum(model.predict_proba(row)[0]), 1, places=6)
        result = self.post("/api/ml/predict", {"practice": 7, "accuracy": 80, "attendance": 90})
        self.assertEqual(result.status_code, 200)
        self.assertIn(result.get_json()["prediction"], {"Қолдау қажет", "Орташа", "Жоғары"})
        self.assertIn("f1", result.get_json()["metrics"])

    def test_classifier_rejects_ranges_types_and_nonfinite_numbers(self):
        for payload in (
            {"practice": 99, "accuracy": 80, "attendance": 90},
            {"practice": True, "accuracy": 80, "attendance": 90},
            {"practice": 5, "accuracy": float("nan"), "attendance": 90},
            {"practice": 5, "accuracy": 80, "attendance": 101},
        ):
            self.assertEqual(self.post("/api/ml/predict", payload).status_code, 400)

    def test_unknown_route_and_invalid_body(self):
        self.assertEqual(local_server.dispatch("GET", "/not-a-route")[0], 404)
        self.assertEqual(local_server.dispatch("POST", "/api/session", ["not", "an", "object"])[0], 400)

    def test_q_learning_update_uses_reward_and_future_value(self):
        session = create_session(4)
        state, next_state = q_state(1, 0), q_state(2, 1)
        session["q"][(2, 2, 0)] = .5
        self.assertAlmostEqual(update_q(session["q"], state, 2, 1, next_state), .4275)

    def test_macro_metrics_are_computed_over_each_class(self):
        actual = ["a", "a", "b", "b"]
        self.assertEqual(_macro_metrics(actual, ["a", "b", "b", "b"], ["a", "b"]), (.75, .833, .75, .733))

    def test_server_reads_host_and_dynamic_port_from_environment(self):
        with patch.dict(os.environ, {"HOST": "0.0.0.0", "PORT": "12345"}):
            self.assertEqual(local_server.get_bind_target(), ("0.0.0.0", 12345))
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(local_server.get_bind_target(), ("127.0.0.1", 5000))

    def test_render_blueprint_uses_standard_library_commands(self):
        blueprint = (Path(__file__).parent / "render.yaml").read_text(encoding="utf-8")
        self.assertIn("buildCommand: python -m py_compile learning_engine.py local_server.py", blueprint)
        self.assertIn("startCommand: python local_server.py", blueprint)
        self.assertIn("value: 0.0.0.0", blueprint)


if __name__ == "__main__":
    unittest.main(verbosity=2)
