# QAIRO Smart AI Study Assistant — Technical Report

**Course:** QAIRU AI Fundamentals (AIF1303)  
**Student(s):** [Complete before submission]  
**Date:** [Complete before submission]

## Summary

QAIRO is a responsive website in Kazakh that demonstrates adaptive practice, probabilistic mastery estimation and classical machine learning. The offline local runtime uses Python's standard library only. It needs no account, API key, internet connection, or package installation. HTML, CSS and JavaScript run in the browser; Python serves the site, validates inputs and performs the calculations.

## Problem and design

The project gives students one place to practise three AI topics from the course and inspect how each algorithm changes or produces a result. A local HTTP server serves the same-origin web UI and JSON API. Learning state is held in process memory only; it is neither saved to disk nor sent to a third party. Reloading or restarting the server creates a fresh session.

## Syllabus topics and implementations

| Week | Topic | In-project implementation | Website demonstration |
|---|---|---|---|
| 5 | Markov decision processes and Q-learning | Tabular Q values, states from difficulty and recent correct-answer streak, reward of +1/−1, learning rate 0.3, discount 0.85, epsilon-greedy selection (epsilon 0.1) and three bounded difficulty actions | Kazakh quiz, answer feedback, next difficulty and Q-value bars |
| 6 | Probability and Bayesian inference | Binary mastery hypothesis; log-odds update using prior, sensitivity, guessing probability and correct/incorrect counts | Editable assumptions and counts, posterior probability and an interpretation |
| 8 | Rational decision making and classical ML | Standardized multinomial logistic regression implemented from scratch; softmax probabilities, cross-entropy gradient descent, L2 regularization and deterministic synthetic data with a stratified 75/25 split | Editable feature values, predicted class and hold-out accuracy, macro precision, recall and F1 |

The classifier is a real trained model, but its dataset is generated for demonstration. The results do not establish validity for real students and must not be used for grading, admission or placement.

## Runtime and repository

`local_server.py` uses `http.server`, `json`, `urllib`, and other Python standard-library modules. `launch.py` starts that server, checks its local health endpoint and opens the default browser. Windows has `start_windows.bat`; macOS/Linux has `start_unix.sh`. The server binds only to `127.0.0.1` by default. Assets and API requests are local; Google Fonts and other external resources are not loaded.

`learning_engine.py` contains the three algorithms and the synthetic training data. `templates/index.html`, `static/css/style.css` and `static/js/app.js` provide the responsive Kazakh interface. `test_app.py` exercises the API dispatcher, static assets, invalid requests, algorithm outputs, and hosting settings without opening a network port. A separate local HTTP smoke check previously verified the running server and key endpoints. Render uses `render.yaml` and `local_server.py`; the legacy `server.py` and `requirements.txt` are optional and are not used by the offline or Render launch paths.

## Bayesian model

The app treats mastery as a binary hypothesis. For each correct answer it adds `log(P(correct | mastered) / P(correct | not mastered))` to prior log odds; incorrect answers use the corresponding incorrect-response likelihood ratio. The resulting odds convert back to a probability. The UI exposes the prior and likelihood assumptions. The output is a model estimate, not a validated score.

## Q-learning model

For each state/action pair the update is `Q(s,a) ← Q(s,a) + α[r + γ maxₐ′Q(s′,a′) − Q(s,a)]`. The state is the clamped difficulty and a small discretized correct-answer streak. The actions lower, maintain or raise difficulty. An epsilon-greedy policy explores with probability 0.1 and otherwise chooses the action with the highest learned Q value (ties prefer maintaining the current challenge). Q values start at zero for each session.

## Classifier and evaluation

The fixed-seed synthetic dataset has practice count, test accuracy and attendance as numeric features and three threshold-derived outcome classes. The implementation standardizes features using training-set means and standard deviations, then fits multinomial logistic regression with softmax, cross-entropy batch gradients and L2 weight regularization. A deterministic stratified split reserves 25% for evaluation. The reported metrics are calculated on this held-out synthetic split; they measure this exercise only. The implementation is transparent and requires no numerical or machine-learning package.

## Validation and errors

The API validates session IDs, topics, question identity, answer indexes, Bayesian probability/count ranges and classifier feature ranges, including non-finite values. Malformed JSON and unknown routes receive JSON error responses. Errors are shown in Kazakh. Test instructions and current test results are recorded in the README; only results actually observed in the current environment should be treated as verified.

## Privacy, ethics and limitations

- No account, credential, API key, external service or student record is used.
- Session data exists only in server memory and is removed when the process ends.
- The Bayesian assumptions are illustrative and affect the posterior.
- The Q-learning environment is small and does not show that the policy is pedagogically optimal.
- The classifier labels and metrics come from synthetic rules and cannot justify conclusions about real students.
- Names, IDs, team contributions, signatures and dates remain editable fields for the student to complete truthfully.

## Running and publishing

Run the offline version with `start_windows.bat` or `py -3 launch.py` on Windows, or `python3 launch.py` on macOS/Linux. No virtual environment is needed. Run `py -3 -m unittest -v` (or the platform's Python command) to execute the test suite.

The local Git repository is committed and prepared for a later GitHub push. `render.yaml` now configures Render to run the standard-library server, bind to `0.0.0.0` and read the assigned `PORT`; its build command compiles Python files and does not install packages. `Procfile` matches that start command. The host settings and Blueprint commands are covered by tests. No GitHub remote or Render account was connected, so the public service and its generated URL have not been created. Free Render web services can spin down when idle; session data lives in memory and resets after restarts.

## Requirement-document status

The referenced conversation and existing project materials identify weeks 5, 6 and 8 as the intended syllabus topics. The two named source documents, `QAIRU_AI_Fundamentals_Final_Project_2026_KZ.docx` and `AI syllabus(1).docx`, were not attached to the referenced conversation and were not found in the accessible project/workspace folders. The exact challenge instructions, grading rubric, required report structure and point weighting therefore remain unchecked. Compare this project with the original documents before submission.
