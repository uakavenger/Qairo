"""Dependency-light local preview server; production deployment uses Flask in server.py."""
from __future__ import annotations

import json
import math
import os
import secrets
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from learning_engine import ACTION_DELTAS, QUESTIONS, bayesian_mastery, choose_q_action, create_session, q_state, train_classifier, update_q

ROOT = Path(__file__).resolve().parent
SESSIONS = {}
MODEL, METRICS = train_classifier()


def dispatch(method: str, path: str, data: dict | None = None):
    data = data or {}
    if not isinstance(data, dict):
        return 400, {"error":"Сұрау пішімі жарамсыз."}
    if method == "GET" and path == "/api/health":
        return 200, {"status":"ok","service":"Ақылды оқу көмекшісі"}
    if method == "GET" and path == "/api/ml/metrics":
        return 200, {"metrics":METRICS,"labels":["Қолдау қажет","Орташа","Жоғары"],"features":["Жаттығу саны","Тест дәлдігі","Қатысу"],"dataset":"Синтетикалық оқу үлгісі"}
    if method == "GET" and path.startswith("/api/analytics"):
        sid=parse_qs(urlparse(path).query).get("session_id",[""])[0]
        s=SESSIONS.get(sid)
        if s is None:return 404,{"error":"Оқу сессиясы табылмады."}
        total=s["correct"]+s["incorrect"]
        return 200,{"total":total,"correct":s["correct"],"incorrect":s["incorrect"],"accuracy":round(s["correct"]/total,3) if total else 0,"mastery":round(s["mastery"],4),"history":s["history"][-12:],"next_difficulty":s["difficulty"]}
    if method == "POST" and path == "/api/session":
        sid=secrets.token_urlsafe(18);SESSIONS[sid]=create_session(seed=int.from_bytes(os.urandom(4),"big"));return 200,{"session_id":sid,"message":"Оқу сессиясы басталды."}
    sid=data.get("session_id");s=SESSIONS.get(sid) if isinstance(sid,str) else None
    if path in ("/api/quiz/next","/api/quiz/answer") and s is None:return 404,{"error":"Оқу сессиясы табылмады. Бетті жаңартып, қайта бастаңыз."}
    if method == "POST" and path == "/api/quiz/next":
        topic=data.get("topic")
        if not isinstance(topic,str) or topic not in QUESTIONS:return 400,{"error":"Тақырыпты тізімнен таңдаңыз."}
        difficulty=s["difficulty"];items=QUESTIONS[topic][difficulty];position=s["topic_index"][topic]%len(items);s["topic_index"][topic]+=1;source=items[position];qid=f"{topic}:{difficulty}:{position}"
        s["current"]={"id":qid,"topic":topic,"difficulty":difficulty,**source}
        return 200,{"id":qid,"topic":topic,"difficulty":difficulty,"difficulty_label":("Бастапқы","Орташа","Күрделі")[difficulty-1],"prompt":source["prompt"],"choices":source["choices"],"question_number":len(s["history"])+1}
    if method == "POST" and path == "/api/quiz/answer":
        current=s["current"]
        if current is None or data.get("question_id")!=current["id"]:return 409,{"error":"Бұл сұрақ ескірген. Келесі сұрақты қайта ашыңыз."}
        choice=data.get("choice")
        if isinstance(choice,bool) or not isinstance(choice,int) or not 0<=choice<len(current["choices"]):return 400,{"error":"Жауап нұсқасын таңдаңыз."}
        correct=choice==current["answer"];s["correct"]+=int(correct);s["incorrect"]+=int(not correct);s["streak"]=min(2,s["streak"]+1) if correct else 0;s["mastery"]=bayesian_mastery(.5,s["correct"],s["incorrect"]);reward=1. if correct else -1.;next_state=q_state(current["difficulty"],s["streak"])
        if s["pending"] is not None:
            old_state,old_action=s["pending"];update_q(s["q"],old_state,old_action,reward,next_state)
        action=choose_q_action(s["q"],next_state,s["rng"]);next_difficulty=max(1,min(3,current["difficulty"]+ACTION_DELTAS[action]));s["pending"]=(next_state,action);s["difficulty"]=next_difficulty;s["history"].append({"topic":current["topic"],"difficulty":current["difficulty"],"correct":correct,"reward":reward,"q_value":s["q"][(next_state[0],next_state[1],action)]})
        return 200,{"correct":correct,"explanation":current["explanation"],"mastery":round(s["mastery"],4),"correct_count":s["correct"],"incorrect_count":s["incorrect"],"reward":reward,"next_difficulty":next_difficulty,"next_difficulty_label":("Бастапқы","Орташа","Күрделі")[next_difficulty-1],"q_table":{"жеңілдету":round(s["q"][(next_state[0],next_state[1],0)],3),"сақтау":round(s["q"][(next_state[0],next_state[1],1)],3),"күрделендіру":round(s["q"][(next_state[0],next_state[1],2)],3)}}
    if method == "POST" and path == "/api/bayes":
        try:
            correct,incorrect=data.get("correct"),data.get("incorrect")
            if isinstance(correct,bool) or not isinstance(correct,int) or isinstance(incorrect,bool) or not isinstance(incorrect,int):raise ValueError("Жауап сандары бүтін сан болуы керек.")
            p=bayesian_mastery(float(data.get("prior",.5)),correct,incorrect,float(data.get("sensitivity",.8)),float(data.get("guess",.25)))
            return 200,{"posterior":round(p,4),"percent":round(p*100,1),"correct":correct,"incorrect":incorrect,"interpretation":"Болжамдық оқу бағасы; ресми баға немесе диагноз емес."}
        except (ValueError,TypeError,OverflowError) as exc:return 400,{"error":str(exc)}
    if method == "POST" and path == "/api/ml/predict":
        try:
            raw_practice=data.get("practice")
            if isinstance(raw_practice,bool) or not isinstance(raw_practice,int):raise ValueError
            practice,accuracy,attendance=raw_practice,float(data.get("accuracy")),float(data.get("attendance"))
            if not 0<=practice<=20 or not math.isfinite(accuracy) or not math.isfinite(attendance) or not 0<=accuracy<=100 or not 0<=attendance<=100:raise ValueError
        except (ValueError,TypeError):return 400,{"error":"Сандарды тексеріңіз: жаттығу саны 0–20, пайыздар 0–100 аралығында болуы керек."}
        row=[[practice,accuracy,attendance]];pred=str(MODEL.predict(row)[0]);confidence=float(max(MODEL.predict_proba(row)[0]));advice={"Жоғары":"Күрделі есептермен біліміңізді тереңдетіңіз.","Орташа":"Қысқа қайталау жасап, қосымша жаттығу орындаңыз.","Қолдау қажет":"Негізгі ұғымдарды қайталап, жеңіл тапсырмадан бастаңыз."}[pred]
        return 200,{"prediction":pred,"confidence":round(confidence,3),"recommendation":advice,"metrics":METRICS,"dataset":"Синтетикалық оқу үлгісі"}
    return 404,{"error":"Сұралған бет табылмады."}


class Handler(BaseHTTPRequestHandler):
    def respond(self,status,body,content_type="application/json; charset=utf-8"):
        if isinstance(body,(dict,list)):body=json.dumps(body,ensure_ascii=False).encode("utf-8")
        self.send_response(status);self.send_header("Content-Type",content_type);self.send_header("Content-Length",str(len(body)));self.send_header("Cache-Control","no-store");self.end_headers();self.wfile.write(body)
    def do_GET(self):
        path=urlparse(self.path).path
        if path=="/":
            html=(ROOT/"templates"/"index.html").read_text(encoding="utf-8").replace("{{ url_for('static', filename='css/style.css') }}","/static/css/style.css").replace("{{ url_for('static', filename='js/app.js') }}","/static/js/app.js")
            return self.respond(200,html.encode("utf-8"),"text/html; charset=utf-8")
        if path.startswith("/static/"):
            asset=(ROOT/path.lstrip("/")).resolve()
            if not str(asset).startswith(str((ROOT/"static").resolve())) or not asset.is_file():return self.respond(404,b"Not found","text/plain")
            ctype="text/css; charset=utf-8" if asset.suffix==".css" else "application/javascript; charset=utf-8"
            return self.respond(200,asset.read_bytes(),ctype)
        status,result=dispatch("GET",self.path);return self.respond(status,result)
    def do_POST(self):
        try:
            length=int(self.headers.get("Content-Length","0"))
            if length < 0 or length > 1_000_000:
                return self.respond(413,{"error":"Сұрау көлемі рұқсат етілген шектен асты."})
            data=json.loads(self.rfile.read(length) or b"{}")
        except (ValueError,UnicodeDecodeError):return self.respond(400,{"error":"Сұрау пішімін тексеріңіз."})
        if not isinstance(data,dict):return self.respond(400,{"error":"Сұрау пішімі жарамсыз."})
        status,result=dispatch("POST",urlparse(self.path).path,data);return self.respond(status,result)
    def log_message(self,format,*args):pass


def main():
    port=int(os.environ.get("PORT","5000"));httpd=ThreadingHTTPServer(("127.0.0.1",port),Handler)
    print(f"QAIRO local preview: http://127.0.0.1:{port}",flush=True);httpd.serve_forever()


if __name__=="__main__":main()
