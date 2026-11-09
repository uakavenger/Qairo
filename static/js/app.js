(() => {
  const $ = (selector) => document.querySelector(selector);
  const $$ = (selector) => [...document.querySelectorAll(selector)];
  const labels = {home:"Басты бет",practice:"Білімді тексеру",insights:"AI талдауы",plan:"Жеке оқу жоспары"};
  const sessionKey = "qairo_session_id";
  let sessionId = sessionStorage.getItem(sessionKey);
  let question = null;
  let totalAnswered = 0;
  let lastMastery = .5;
  let pendingTopic = null;
  const toast = (message) => { const el=$("#toast"); el.textContent=message; el.classList.add("show"); setTimeout(()=>el.classList.remove("show"),2300); };
  async function api(path, body, method="POST") {
    const response = await fetch(path, {method, headers:{"Content-Type":"application/json"}, body: body ? JSON.stringify(body) : undefined});
    let data; try { data=await response.json(); } catch { throw new Error("Сервер жауабын оқу мүмкін болмады."); }
    if (!response.ok) throw new Error(data.error || "Сұрауды орындау мүмкін болмады.");
    return data;
  }
  async function ensureSession(reset=false) {
    if (reset) { sessionStorage.removeItem(sessionKey); sessionId=null; }
    if (!sessionId) { const data=await api("/api/session",{}); sessionId=data.session_id; sessionStorage.setItem(sessionKey,sessionId); }
    return sessionId;
  }
  function setSection(name) {
    $$(".page-section").forEach(el=>el.classList.toggle("active",el.id===`section-${name}`));
    $$(".nav-link").forEach(el=>el.classList.toggle("active",el.dataset.section===name));
    $("#crumb").textContent=labels[name]||labels.home;
    $("#sidebar").classList.remove("open");
    window.scrollTo({top:0,behavior:"smooth"});
    if(name==="insights") loadMetrics();
    if(name==="plan") refreshPlan();
  }
  function difficultyName(n){return ["Бастапқы","Орташа","Күрделі"][n-1]||"Бастапқы";}
  async function loadQuestion(topic=$("#topicSelect").value) {
    try {
      await ensureSession(); pendingTopic=topic;
      const data=await api("/api/quiz/next",{session_id:sessionId,topic}); question=data;
      $("#difficultyChip").textContent=`${data.difficulty_label} деңгей`;
      $("#questionText").textContent=data.prompt;
      $("#questionNumber").textContent=`Сұрақ ${data.question_number}`;
      $("#quizProgress").style.width=`${Math.min(100,((data.question_number-1)%8+1)*12.5)}%`;
      $("#answerChoices").replaceChildren(...data.choices.map((choice,index)=>{
        const label=document.createElement("label"); label.className="answer-option";
        const input=document.createElement("input"); input.type="radio"; input.name="answer"; input.value=index; input.required=true;
        input.addEventListener("change",()=>{$$(".answer-option").forEach(el=>el.classList.remove("selected"));label.classList.add("selected");});
        const span=document.createElement("span"); span.textContent=choice; label.append(input,span); return label;
      }));
      $("#answerButton").disabled=false;
      $("#answerFeedback").hidden=true;
    } catch(err){toast(err.message);}
  }
  function renderQ(q) {
    const keys=[["жеңілдету","qDown","qDownValue"],["сақтау","qStay","qStayValue"],["күрделендіру","qUp","qUpValue"]];
    keys.forEach(([key,bar,value])=>{const v=q[key]||0; $("#"+value).textContent=v.toFixed(2); $("#"+bar).style.width=`${Math.max(3,Math.min(100,(v+1)*45))}%`;});
  }
  function updateHome(data) {
    totalAnswered=data.total||totalAnswered; lastMastery=data.mastery??lastMastery;
    $("#homeMastery").innerHTML=`${Math.round(lastMastery*100)}<span>%</span>`;
    $("#homeMasteryBar").style.width=`${lastMastery*100}%`;
    $("#homeQuestionCount").textContent=data.total||0;
    $("#homeNextLevel").textContent=difficultyName(data.next_difficulty||1);
    $("#quizCount").textContent=data.total||0;
    $("#planLevel").textContent=(data.total||0)?`${Math.round(lastMastery*100)}% · меңгеру бағасы`:"Әзірге дерек жеткіліксіз";
    $("#planSummary").textContent=(data.total||0)?`${data.correct||0} дұрыс жауап, ${data.incorrect||0} қате жауап. Келесі тапсырма деңгейі жауаптарыңызға қарай өзгеріп отырады.`:"Бірнеше тест сұрағына жауап беріп, талдауды бастаңыз.";
  }
  async function loadAnalytics(){
    if(!sessionId)return;
    try { const data=await api(`/api/analytics?session_id=${encodeURIComponent(sessionId)}`,null,"GET"); updateHome(data); } catch {}
  }
  async function refreshPlan(){await loadAnalytics();}
  async function loadMetrics(){
    try {const data=await api("/api/ml/metrics",null,"GET");
      const m=data.metrics; [["accuracy","metricAccuracy"],["precision","metricPrecision"],["recall","metricRecall"],["f1","metricF1"]].forEach(([key,id])=>$("#"+id).textContent=m[key].toFixed(2));
      const chart=$("#metricChart"); chart.replaceChildren(...[["Дәлдік",m.accuracy],["Оң болжам",m.precision],["Толықтық",m.recall],["F1 өлшемі",m.f1]].map(([name,value])=>{const bar=document.createElement("div");bar.className="metric-bar";bar.style.height=`${Math.max(4,value*80)}px`;const label=document.createElement("span");label.textContent=name;bar.append(label);return bar;}));
    }catch(err){toast(err.message);}
  }
  $$("[data-section]").forEach(btn=>btn.addEventListener("click",()=>{if(btn.dataset.topic)$("#topicSelect").value=btn.dataset.topic;setSection(btn.dataset.section);if(btn.dataset.section==="practice")loadQuestion(btn.dataset.topic||$("#topicSelect").value);if(btn.dataset.algorithm)selectAlgorithm(btn.dataset.algorithm);}));
  $$("[data-goto]").forEach(btn=>btn.addEventListener("click",()=>{if(btn.dataset.topic)$("#topicSelect").value=btn.dataset.topic;setSection(btn.dataset.goto);if(btn.dataset.goto==="practice")loadQuestion(btn.dataset.topic||$("#topicSelect").value);if(btn.dataset.algorithm)selectAlgorithm(btn.dataset.algorithm);}));
  function selectAlgorithm(which){$$(".algorithm-tab").forEach(tab=>tab.classList.toggle("active",tab.dataset.tab===which));$$(".algorithm-pane").forEach(pane=>pane.classList.toggle("active",pane.id===`pane-${which}`));}
  $$(".algorithm-tab").forEach(tab=>tab.addEventListener("click",()=>selectAlgorithm(tab.dataset.tab)));
  $("#menuToggle").addEventListener("click",()=>$("#sidebar").classList.toggle("open"));
  $("#topicSelect").addEventListener("change",()=>loadQuestion());
  $("#answerForm").addEventListener("submit",async event=>{
    event.preventDefault();const selected=$("input[name=answer]:checked");if(!selected){toast("Алдымен жауап нұсқасын таңдаңыз.");return;} if(!question)return;
    const button=$("#answerButton");button.disabled=true;
    try {const result=await api("/api/quiz/answer",{session_id:sessionId,question_id:question.id,choice:Number(selected.value)});const feedback=$("#answerFeedback");feedback.hidden=false;feedback.className=`feedback ${result.correct?"good":"bad"}`;feedback.textContent=`${result.correct?"Дұрыс жауап!":"Қате жауап."} ${result.explanation} Келесі сұрақ: ${result.next_difficulty_label.toLowerCase()} деңгей.`;renderQ(result.q_table);$("#homeNextLevel").textContent=result.next_difficulty_label;totalAnswered+=1;lastMastery=result.mastery;$("#homeMastery").innerHTML=`${Math.round(lastMastery*100)}<span>%</span>`;$("#homeMasteryBar").style.width=`${lastMastery*100}%`;$("#homeQuestionCount").textContent=totalAnswered;$("#quizCount").textContent=totalAnswered;$("#questionNumber").textContent=`${totalAnswered} жауап берілді`;$("#planLevel").textContent=`${Math.round(lastMastery*100)}% · меңгеру бағасы`;$("#planSummary").textContent=`${result.correct_count} дұрыс жауап, ${result.incorrect_count} қате жауап. Қайталап жаттығып, нәтижені жақсартыңыз.`;setTimeout(()=>loadQuestion(pendingTopic||$("#topicSelect").value),1350);}
    catch(err){button.disabled=false;toast(err.message);}
  });
  $("#newSession").addEventListener("click",async()=>{await ensureSession(true);totalAnswered=0;lastMastery=.5;renderQ({"жеңілдету":0,"сақтау":0,"күрделендіру":0});updateHome({total:0,mastery:.5,next_difficulty:1,correct:0,incorrect:0});await loadQuestion($("#topicSelect").value);toast("Жаңа оқу сессиясы басталды.");});
  $("#priorInput").addEventListener("input",event=>$("#priorValue").textContent=`${event.target.value}%`);
  $("#bayesForm").addEventListener("submit",async event=>{event.preventDefault();const error=$("#bayesError");error.hidden=true;try{const result=await api("/api/bayes",{prior:Number($("#priorInput").value)/100,correct:Number($("#bayesCorrect").value),incorrect:Number($("#bayesIncorrect").value),sensitivity:Number($("#sensitivity").value),guess:Number($("#guess").value)});$("#posteriorPercent").textContent=`${result.percent}%`;$("#posteriorRing").style.background=`conic-gradient(#7865e9 ${result.percent*3.6}deg,#e3e0fa 0deg)`;$("#posteriorCaption").textContent=result.percent>=75?"Тақырып жақсы меңгерілген сияқты.":result.percent>=45?"Білім біртіндеп қалыптасып келеді.":"Негізгі ұғымдарды қайталау ұсынылады.";toast("Байес есебі серверде орындалды.");}catch(err){error.hidden=false;error.textContent=err.message;}});
  $("#mlForm").addEventListener("submit",async event=>{event.preventDefault();const error=$("#mlError");error.hidden=true;try{const result=await api("/api/ml/predict",{practice:Number($("#mlPractice").value),accuracy:Number($("#mlAccuracy").value),attendance:Number($("#mlAttendance").value)});$("#predictionLabel").textContent=result.prediction;$("#predictionAdvice").textContent=result.recommendation;$("#confidenceValue").textContent=`${Math.round(result.confidence*100)}%`;$("#confidenceBar").style.width=`${result.confidence*100}%`;loadMetrics();toast("ML болжамы Python backend арқылы есептелді.");}catch(err){error.hidden=false;error.textContent=err.message;}});
  (async()=>{try{await ensureSession();await loadQuestion();await loadAnalytics();}catch(err){toast(err.message);}})();
})();
