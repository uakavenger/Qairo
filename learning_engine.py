"""Server-side AI teaching algorithms for Smart AI Study Assistant."""
from __future__ import annotations

import math
import random
from collections import defaultdict



QUESTIONS = {
    "Q-learning": {
        1: [
            {"prompt": "Q-learning агенті нені бағалайды?", "choices": ["Күй мен әрекеттің ұзақ мерзімді құнын", "Тек сұрақ мәтінін", "Деректердің түсін", "Кездейсоқ жауапты"], "answer": 0, "explanation": "Q(s,a) күй s кезіндегі әрекет a таңдаудың болашақ құнын бағалайды."},
            {"prompt": "MDP-де сыйақы нені көрсетеді?", "choices": ["Агенттің әрекеттен алған кері байланысын", "Сұрақтың ұзындығын", "Күйдің атауын", "Деректер санын"], "answer": 0, "explanation": "Сыйақы агентке әрекет нәтижесінің пайдалы не тиімсіз болғанын көрсетеді."},
        ],
        2: [
            {"prompt": "Q-learning жаңартуында γ коэффициенті нені реттейді?", "choices": ["Болашақ сыйақының салмағын", "Сұрақтар санын", "Күйлердің атын", "Деректер түрін"], "answer": 0, "explanation": "γ болашақтағы сыйақылардың ағымдағы бағалауға ықпалын реттейді."},
            {"prompt": "Эпсилон-greedy саясатының мақсаты қандай?", "choices": ["Барлау мен тиімді әрекетті тең ұстау", "Деректерді жою", "Тек кездейсоқ таңдау", "Жауапты алдын ала көрсету"], "answer": 0, "explanation": "Эпсилон ықтималдығымен жаңа әрекет зерттеледі, қалған кезде ең жоғары Q мәні бар әрекет таңдалады."},
        ],
        3: [
            {"prompt": "Егер оқушы күрделі сұраққа қате жауап берсе, бейімделетін агент не істей алады?", "choices": ["Келесі сұрақты жеңілдету", "Барлық нәтижені өшіру", "Деңгейді өзгеріссіз мәңгі қалдыру", "Жауапты өзі толтыру"], "answer": 0, "explanation": "Сыйақы мен Q мәндері саясатты бейімдейді; жүйе келесіде жеңілірек әрекетті таңдай алады."},
        ],
    },
    "Байес қорытындысы": {
        1: [
            {"prompt": "Байес ережесі жаңа дәлел түскенде нені жаңартады?", "choices": ["Априорлық сенімді", "Экран өлшемін", "Файл атын", "Сұрақ ретін"], "answer": 0, "explanation": "Байес ережесі априорлық ықтималдықты дәлелдің ықтималдығымен жаңартады."},
            {"prompt": "P(A|B) қандай мағына береді?", "choices": ["B шарты берілгендегі A ықтималдығы", "A мен B қосындысы", "B-нің қарсы оқиғасы", "A-ның дәл мәні"], "answer": 0, "explanation": "Бұл B шарты орындалған жағдайда A оқиғасының шартты ықтималдығы."},
        ],
        2: [
            {"prompt": "Оқушы дұрыс жауап бергенде меңгеру ықтималдығы әдетте қалай өзгереді?", "choices": ["Өседі", "Міндетті түрде нөл болады", "Өзгермейді", "Теріс болады"], "answer": 0, "explanation": "Берілген модельде дұрыс жауап меңгерілген тақырып гипотезасына көбірек сәйкес келеді."},
            {"prompt": "Байес бағасы қандай екі негізгі ақпаратты біріктіреді?", "choices": ["Априорлық ықтималдық пен дәлел ықтималдығын", "Экран мен желі жылдамдығын", "Жол мен файл көлемін", "Код пен түсті"], "answer": 0, "explanation": "Posterior баға априорлық сенім мен алынған дәлелге сүйенеді."},
        ],
        3: [
            {"prompt": "Байес бағасы неге нақты диагноз емес?", "choices": ["Ол априорлық және likelihood жорамалдарына тәуелді", "Ол ешбір жауапты қолданбайды", "Ол әрқашан 100% дәл", "Ол дерекке тәуелсіз"], "answer": 0, "explanation": "Қолданылған ықтималдық жорамалдары қате болса, нәтиже де өзгереді."},
        ],
    },
    "Классикалық ML": {
        1: [
            {"prompt": "Жіктеу моделі нені болжайды?", "choices": ["Санатты", "Тек абзац ұзындығын", "Компьютер түрін", "Кездейсоқ түсті"], "answer": 0, "explanation": "Жіктеу кіріс белгілеріне сүйеніп санаттық белгіні болжайды."},
            {"prompt": "Оқыту және тест деректерін бөлу не үшін қажет?", "choices": ["Көрмеген деректегі жалпылауды бағалау", "Баған атауларын жасыру", "Деректерді жоғалту", "Модельді өшіру"], "answer": 0, "explanation": "Тест жиыны модельдің оқытуда пайдаланылмаған деректегі сапасын бағалайды."},
        ],
        2: [
            {"prompt": "Precision метрикасы нені өлшейді?", "choices": ["Оң деп белгіленгендердің қаншасы дұрыс екенін", "Барлық белгілердің орташа мәнін", "Оқыту уақытын", "Деректер көлемін"], "answer": 0, "explanation": "Precision — болжанған оң мысалдардың ішіндегі шын оңдардың үлесі."},
            {"prompt": "Логистикалық регрессия әдетте қандай есептерде қолданылады?", "choices": ["Жіктеу", "Сурет салу", "Мәтінді сақтау", "Кестені форматтау"], "answer": 0, "explanation": "Логистикалық регрессия кіріс белгілерінен санат ықтималдығын бағалай алады."},
        ],
        3: [
            {"prompt": "Синтетикалық дерек нәтижесін шынайы студенттерге бірден қолдануға бола ма?", "choices": ["Жоқ, нақты деректе бөлек тексеру қажет", "Иә, ол әрдайым мінсіз", "Иә, тест қажет емес", "Тек атауын өзгерту жеткілікті"], "answer": 0, "explanation": "Синтетикалық демонстрация нақты студент дерегіндегі сапаны дәлелдемейді."},
        ],
    },
}

ACTION_DELTAS = (-1, 0, 1)  # жеңілдету, сақтау, күрделендіру


def bayesian_mastery(prior: float, correct: int, incorrect: int, sensitivity: float = .8, guess: float = .25) -> float:
    if not 0 < prior < 1:
        raise ValueError("Бастапқы ықтималдық 0 мен 1 арасында болуы керек.")
    if isinstance(correct, bool) or isinstance(incorrect, bool) or correct < 0 or incorrect < 0 or correct + incorrect == 0 or correct > 100 or incorrect > 100:
        raise ValueError("Әр жауап саны 0–100 аралығында болуы керек және жалпы саны нөлден көп болуы қажет.")
    if not 0 < sensitivity < 1 or not 0 < guess < 1:
        raise ValueError("Модель ықтималдықтары 0 мен 1 арасында болуы керек.")
    log_odds = math.log(prior / (1 - prior))
    log_odds += correct * math.log(sensitivity / guess)
    log_odds += incorrect * math.log((1 - sensitivity) / (1 - guess))
    if log_odds >= 0:
        return 1 / (1 + math.exp(-min(log_odds, 700)))
    exp_value = math.exp(max(log_odds, -700))
    return exp_value / (1 + exp_value)


def q_state(difficulty: int, streak: int) -> tuple[int, int]:
    return (max(1, min(3, difficulty)), max(0, min(2, streak + 1)))


def choose_q_action(q_values: dict, state: tuple[int, int], rng: random.Random, epsilon: float = .1) -> int:
    if rng.random() < epsilon:
        return rng.randrange(3)
    values = [q_values[(state[0], state[1], action)] for action in range(3)]
    best = max(values)
    # deterministic tie break favors maintaining the current challenge
    return next(i for i, value in enumerate(values) if value == best)


def update_q(q_values: dict, state: tuple[int, int], action: int, reward: float, next_state: tuple[int, int], alpha: float = .3, gamma: float = .85) -> float:
    if action not in (0, 1, 2) or not -1 <= reward <= 1:
        raise ValueError("Q-learning әрекеті немесе сыйақы мәні жарамсыз.")
    old = q_values[(state[0], state[1], action)]
    future = max(q_values[(next_state[0], next_state[1], a)] for a in range(3))
    new = old + alpha * (reward + gamma * future - old)
    q_values[(state[0], state[1], action)] = new
    return new


def make_training_data(seed: int = 21):
    rng = random.Random(seed)
    x, y = [], []
    for _ in range(240):
        practice = rng.randint(0, 10)
        accuracy = max(0, min(100, round(22 + practice * 6 + rng.gauss(0, 12))))
        attendance = max(0, min(100, round(35 + practice * 5 + rng.gauss(0, 15))))
        score = .55 * accuracy + .25 * attendance + 2 * practice
        label = "Жоғары" if score >= 67 else "Орташа" if score >= 45 else "Қолдау қажет"
        x.append([practice, accuracy, attendance])
        y.append(label)
    return x, y


class LogisticModel:
    """Small from-scratch multinomial logistic regression used if scikit-learn is unavailable."""
    def __init__(self, means, scales, classes, weights):
        self.means, self.scales, self.classes, self.weights = means, scales, classes, weights

    def _scores(self, row):
        x = [(value - mean) / scale for value, mean, scale in zip(row, self.means, self.scales)] + [1.0]
        scores = [sum(w * v for w, v in zip(weights, x)) for weights in self.weights]
        pivot = max(scores)
        exps = [math.exp(max(-50, min(50, score - pivot))) for score in scores]
        total = sum(exps)
        return [value / total for value in exps]

    def predict_proba(self, rows):
        return [self._scores(row) for row in rows]

    def predict(self, rows):
        return [self.classes[max(range(len(p)), key=p.__getitem__)] for p in self.predict_proba(rows)]


def _macro_metrics(actual, predicted, classes):
    accuracy = sum(a == p for a, p in zip(actual, predicted)) / len(actual)
    precisions, recalls, f1s = [], [], []
    for label in classes:
        tp = sum(a == label and p == label for a, p in zip(actual, predicted))
        fp = sum(a != label and p == label for a, p in zip(actual, predicted))
        fn = sum(a == label and p != label for a, p in zip(actual, predicted))
        precision = tp / (tp + fp) if tp + fp else 0
        recall = tp / (tp + fn) if tp + fn else 0
        precisions.append(precision); recalls.append(recall)
        f1s.append(2 * precision * recall / (precision + recall) if precision + recall else 0)
    return round(accuracy, 3), round(sum(precisions)/len(classes), 3), round(sum(recalls)/len(classes), 3), round(sum(f1s)/len(classes), 3)


def _fit_logistic(x_train, y_train, classes):
    means = [sum(row[j] for row in x_train) / len(x_train) for j in range(3)]
    scales = []
    for j, mean in enumerate(means):
        variance = sum((row[j]-mean)**2 for row in x_train)/len(x_train)
        scales.append(math.sqrt(variance) or 1)
    matrix = [[(v-m)/s for v,m,s in zip(row,means,scales)] + [1.0] for row in x_train]
    weights = [[0.0] * 4 for _ in classes]
    # Batch gradient descent on multinomial cross entropy with mild L2 regularization.
    lr, regularization, epochs = .08, .001, 700
    for _ in range(epochs):
        grads = [[0.0] * 4 for _ in classes]
        for row, label in zip(matrix, y_train):
            scores = [sum(w*v for w,v in zip(ws,row)) for ws in weights]
            pivot = max(scores); expv = [math.exp(max(-50,min(50,s-pivot))) for s in scores]; den = sum(expv)
            probs = [v/den for v in expv]
            target = classes.index(label)
            for c in range(len(classes)):
                err = probs[c] - int(c == target)
                for j, feature in enumerate(row): grads[c][j] += err * feature
        for c in range(len(classes)):
            for j in range(4):
                grad = grads[c][j]/len(matrix) + (regularization*weights[c][j] if j < 3 else 0)
                weights[c][j] -= lr*grad
    return LogisticModel(means, scales, classes, weights)


def train_classifier(seed: int = 21, prefer_sklearn: bool = False):
    x, y = make_training_data(seed)
    rng = random.Random(seed)
    classes = sorted(set(y))
    train_idx, test_idx = [], []
    for label in classes:
        indexes = [i for i, value in enumerate(y) if value == label]
        rng.shuffle(indexes)
        split = max(1, round(len(indexes)*.75))
        train_idx.extend(indexes[:split]); test_idx.extend(indexes[split:])
    # The default is the transparent, in-project implementation so the website
    # behaves the same offline and does not need package installation.
    if prefer_sklearn:
      try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
        from sklearn.model_selection import train_test_split
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.25, random_state=seed, stratify=y)
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500, random_state=seed))
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)
        metrics = {"accuracy": round(float(accuracy_score(y_test,predictions)),3), "precision": round(float(precision_score(y_test,predictions,average="macro",zero_division=0)),3), "recall": round(float(recall_score(y_test,predictions,average="macro",zero_division=0)),3), "f1": round(float(f1_score(y_test,predictions,average="macro",zero_division=0)),3), "train_size":len(x_train), "test_size":len(x_test)}
        return model, metrics
      except ImportError:
        pass
    x_train = [x[i] for i in train_idx]; y_train = [y[i] for i in train_idx]
    x_test = [x[i] for i in test_idx]; y_test = [y[i] for i in test_idx]
    model = _fit_logistic(x_train, y_train, classes)
    p = model.predict(x_test)
    accuracy, precision, recall, f1 = _macro_metrics(y_test, p, classes)
    metrics = {"accuracy":accuracy,"precision":precision,"recall":recall,"f1":f1,"train_size":len(x_train),"test_size":len(x_test)}
    return model, metrics


def create_session(seed: int) -> dict:
    return {"rng": random.Random(seed), "q": defaultdict(float), "difficulty": 1, "streak": 0, "mastery": .5, "correct": 0, "incorrect": 0, "current": None, "pending": None, "topic_index": defaultdict(int), "history": []}

