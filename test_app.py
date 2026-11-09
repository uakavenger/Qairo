"""Offline tests for the standard-library website and its learning algorithms."""
import json
import threading
import unittest
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import local_server
from learning_engine import (
    QUESTIONS, _macro_metrics, bayesian_mastery, create_session, make_training_data,
    q_state, train_classifier, update_q,
)


class WebsiteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.httpd = ThreadingHTTPServer(("127.0.0.1", 0), local_server.Handler)
        cls.port = cls.httpd.server_address[1]
        cls.thread = threading.Thread(target=cls.httpd.serve_forever, daemon=True)
        cls.thread.start()
        cls.base = f"http://127.0.0.1:{cls.port}"

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()
        cls.thread.join(timeout=2)

    def request(self, path, method="GET", payload=None, raw=None):
        body = raw if raw is not None else (json.dumps(payload).encode() if payload is not None else None)
        req = Request(self.base + path, data=body, method=method,
                      headers={"Content-Type": "application/json"} if body is not None else {})
        try:
            with urlopen(req, timeout=3) as response:
                content = response.read()
                return response.status, content, response.headers.get_content_type()
        except HTTPError as exc:
            return exc.code, exc.read(), exc.headers.get_content_type()

    def post(self, path, payload):
        status, body, _ = self.request(path, "POST", payload)
        return status, json.loads(body)

    def new_session(self):
        status, result = self.post("/api/session", {})
        self.assertEqual(status, 200)
        return result["session_id"]

    def test_home_and_assets_are_served_without_external_resources(self):
        status, body, content_type = self.request("/")
        html = body.decode("utf-8")
        self.assertEqual((status, content_type), (200, "text/html"))
        self.assertIn('<html lang="kk">', html)
        self.assertIn("Басты бет", html)
        self.assertNotIn("fonts.googleapis.com", html)
        css_status, css, css_type = self.request("/static/css/style.css")
        js_status, js, js_type = self.request("/static/js/app.js")
        self.assertEqual((css_status, css_type), (200, "text/css"))
        self.assertEqual((js_status, js_type), (200, "application/javascript"))
        self.assertIn(b"@media", css)
        self.assertIn(b"/api/quiz/answer", js)

    def test_health_and_metrics_routes(self):
        status, body, _ = self.request("/api/health")
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)["status"], "ok")
        status, body, _ = self.request("/api/ml/metrics")
        result = json.loads(body)
        self.assertEqual(status, 200)
        self.assertGreaterEqual(result["metrics"]["test_size"], 58)
        self.assertLessEqual(result["metrics"]["test_size"], 61)
        self.assertTrue(0 <= result["metrics"]["f1"] <= 1)

    def test_session_quiz_q_learning_and_analytics_flow(self):
        sid = self.new_session()
        status, q = self.post("/api/quiz/next", {"session_id": sid, "topic": "Q-learning"})
        self.assertEqual(status, 200)
        self.assertIn("prompt", q)
        status, result = self.post("/api/quiz/answer", {"session_id": sid, "question_id": q["id"], "choice": 0})
        self.assertEqual(status, 200)
        self.assertTrue(result["correct"])
        self.assertIn("күрделендіру", result["q_table"])
        status, body, _ = self.request("/api/analytics?session_id=" + sid)
        analytics = json.loads(body)
        self.assertEqual(status, 200)
        self.assertEqual((analytics["total"], analytics["correct"]), (1, 1))

    def test_quiz_validates_topic_session_choice_and_stale_question(self):
        sid = self.new_session()
        self.assertEqual(self.post("/api/quiz/next", {"session_id": sid, "topic": []})[0], 400)
        self.assertEqual(self.post("/api/quiz/next", {"session_id": "missing", "topic": "Q-learning"})[0], 404)
        _, q = self.post("/api/quiz/next", {"session_id": sid, "topic": "Q-learning"})
        self.assertEqual(self.post("/api/quiz/answer", {"session_id": sid, "question_id": q["id"], "choice": 99})[0], 400)
        self.assertEqual(self.post("/api/quiz/answer", {"session_id": sid, "question_id": "stale", "choice": 0})[0], 409)

    def test_bayesian_endpoint_and_validation(self):
        status, result = self.post("/api/bayes", {"prior": .5, "correct": 4, "incorrect": 1})
        self.assertEqual(status, 200)
        self.assertGreater(result["posterior"], .5)
        self.assertEqual(self.post("/api/bayes", {"prior": .5, "correct": 0, "incorrect": 0})[0], 400)
        self.assertEqual(self.post("/api/bayes", {"prior": 1.4, "correct": 1, "incorrect": 0})[0], 400)
        self.assertGreater(bayesian_mastery(.5, 1, 0), .5)
        self.assertLess(bayesian_mastery(.5, 0, 1), .5)

    def test_classifier_is_local_deterministic_and_returns_predictions(self):
        x1, y1 = make_training_data()
        x2, y2 = make_training_data()
        self.assertEqual((x1, y1), (x2, y2))
        model, metrics = train_classifier()
        self.assertEqual(metrics["test_size"] + metrics["train_size"], 240)
        self.assertGreaterEqual(metrics["test_size"], 58)
        self.assertLessEqual(metrics["test_size"], 61)
        self.assertEqual(len(model.predict([[7, 80, 90]])), 1)
        self.assertAlmostEqual(sum(model.predict_proba([[7, 80, 90]])[0]), 1, places=6)
        status, result = self.post("/api/ml/predict", {"practice": 7, "accuracy": 80, "attendance": 90})
        self.assertEqual(status, 200)
        self.assertIn(result["prediction"], {"Қолдау қажет", "Орташа", "Жоғары"})
        self.assertIn("f1", result["metrics"])

    def test_classifier_rejects_ranges_types_and_nonfinite_numbers(self):
        for payload in (
            {"practice": 99, "accuracy": 80, "attendance": 90},
            {"practice": True, "accuracy": 80, "attendance": 90},
            {"practice": 5, "accuracy": float("nan"), "attendance": 90},
            {"practice": 5, "accuracy": 80, "attendance": 101},
        ):
            self.assertEqual(self.post("/api/ml/predict", payload)[0], 400)

    def test_malformed_json_unknown_route_and_path_traversal(self):
        self.assertEqual(self.request("/api/session", "POST", raw=b"{")[0], 400)
        self.assertEqual(self.request("/no-such-page")[0], 404)
        self.assertEqual(self.request("/static/../../server.py")[0], 404)

    def test_q_learning_update_uses_reward_and_future_value(self):
        session = create_session(4)
        state, next_state = q_state(1, 0), q_state(2, 1)
        session["q"][(2, 2, 0)] = .5
        value = update_q(session["q"], state, 2, 1, next_state)
        self.assertAlmostEqual(value, .4275)  # 0 + .3 * (1 + .85 * .5)

    def test_macro_metrics_are_computed_over_each_class(self):
        actual = ["a", "a", "b", "b"]
        self.assertEqual(_macro_metrics(actual, ["a", "b", "b", "b"], ["a", "b"]), (.75, .833, .75, .733))


if __name__ == "__main__":
    unittest.main(verbosity=2)
