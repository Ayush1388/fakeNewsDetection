const $ = (id) => document.getElementById(id);
const el = (tag, cls, text) => { const n = document.createElement(tag); if (cls) n.className = cls; if (text != null) n.textContent = text; return n; };
const pct = (p) => Math.round(p * 100);
// Never round a near-certain score up to 100 or a tiny one down to 0: that overstates what the model knows.
const pctText = (p) => (p > 0 && p < 0.005 ? "<1" : p < 1 && p > 0.995 ? ">99" : String(Math.round(p * 100)));

const STAMPS = {
  "true":      { cls: "true",       word: "Verified",     blurb: "reads as a true rumour" },
  "false":     { cls: "false",      word: "False",        blurb: "reads as a false rumour" },
  "unverified":{ cls: "unverified", word: "Unverified",   blurb: "reads as an unverified rumour" },
  "non-rumor": { cls: "non-rumor",  word: "Not a rumour", blurb: "does not read as a rumour" },
};
const VIEW_NAMES = { text: "Text view", spreaders: "Spreader view", cascade: "Cascade view" };
const VIEW_HINT = {
  text: "Judged by the wording alone.",
  spreaders: "Judged by which accounts shared it.",
  cascade: "Judged by how big and how fast the spread was.",
};
const VIEWS = ["intake", "working", "verdict", "examples", "about"];
const NAV_FOR = { intake: "intake", working: "intake", verdict: "intake", examples: "examples", about: "about" };

function show(view) {
  VIEWS.forEach((v) => { $("v-" + v).hidden = v !== view; });
  document.querySelectorAll(".navbtn").forEach((b) => {
    if (b.dataset.view === NAV_FOR[view]) b.setAttribute("aria-current", "page"); else b.removeAttribute("aria-current");
  });
  window.scrollTo(0, 0);
  $("main").focus({ preventScroll: true });
}

function showError(msg) { const e = $("error"); e.textContent = msg || ""; e.hidden = !msg; }

function parseProp(raw) {
  const nodes = [];
  raw.split("\n").forEach((line, i) => {
    if (!line.trim()) return;
    const parts = line.split(/[,\s]+/).filter(Boolean);
    const d = Number(parts[1]);
    if (parts.length < 2 || !Number.isFinite(d) || d < 0)
      throw new Error(`Spread data, line ${i + 1}: write a user id and the minutes after the tweet, like "39364684, 0.45".`);
    nodes.push({ user: parts[0], delay_minutes: d });
  });
  return nodes;
}

async function api(path, opts) {
  let res;
  try { res = await fetch(path, opts); }
  catch { throw new Error("Cannot reach the server. Check that it is running, then try again."); }
  if (!res.ok) {
    let detail = res.statusText;
    try { const j = await res.json(); detail = typeof j.detail === "string" ? j.detail : "The request was not valid."; } catch {}
    throw new Error(detail);
  }
  return res.json();
}

/* ---------- working state ---------- */
function setSteps(state) {
  document.querySelectorAll("#steps li").forEach((li) => {
    li.className = state[li.dataset.step] || "";
  });
}

async function runCheck(body, claimText, hasSpread) {
  $("w-claim").textContent = claimText;
  $("w-note").textContent = "The first check after the server starts trains the model, which can take a minute or two.";
  const state = { text: "active", spreaders: hasSpread ? "active" : "skip", cascade: hasSpread ? "active" : "skip", stack: "" };
  setSteps(state);
  show("working");
  try {
    const r = await api("/api/predict", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    setSteps({ text: "done", spreaders: hasSpread ? "done" : "skip", cascade: hasSpread ? "done" : "skip", stack: "done" });
    render(r, claimText);
    show("verdict");
    const s = $("stamp"); s.classList.remove("land"); void s.offsetWidth; s.classList.add("land");
  } catch (e) {
    show("intake");
    showError(e.message);
  } finally { $("go").disabled = false; }
}

/* ---------- verdict ---------- */
function bars(container, probs, color) {
  container.textContent = "";
  const entries = Object.entries(probs).sort((a, b) => b[1] - a[1]);
  entries.forEach(([k, v], i) => {
    const row = el("div", "bar" + (i === 0 ? " top" : ""));
    if (i === 0 && color) row.style.setProperty("--c", color);
    const track = el("div", "track"); const fill = el("div", "fill"); fill.style.width = (v * 100).toFixed(1) + "%"; track.append(fill);
    row.append(el("span", "", STAMPS[k] ? STAMPS[k].word : k), track, el("span", "v", pctText(v) + "%"));
    container.append(row);
  });
}

function fmtMin(m) {
  if (m < 1) return Math.round(m * 60) + " s";
  if (m < 120) return (m < 10 ? m.toFixed(1) : Math.round(m)) + " min";
  if (m < 2880) return (m / 60).toFixed(1) + " h";
  return (m / 1440).toFixed(1) + " days";
}

function render(r, claimText) {
  const st = STAMPS[r.label];
  $("v-claim").textContent = claimText;
  $("v-meta").textContent = `Model trained on ${r.dataset === "twitter15" ? "Twitter15" : "Twitter16"}`;
  const stamp = $("stamp"); stamp.className = "stamp " + st.cls; stamp.textContent = st.word;
  stamp.setAttribute("aria-label", "Ruling: " + st.word);
  $("v-conf").textContent = pctText(r.confidence);

  const tops = Object.entries(r.per_view).map(([n, p]) => [n, Object.entries(p).sort((a, b) => b[1] - a[1])[0]]);
  const allAgree = tops.every(([, [k]]) => k === r.label);
  $("v-summary").textContent = r.views_used.length === 1
    ? `Text only: this tweet ${st.blurb}. Add spread data for a stronger read.`
    : allAgree ? `All three views agree that this tweet ${st.blurb}.`
    : `This tweet ${st.blurb}, but the views do not all agree. Open each finding to see why.`;

  const f = $("findings"); f.textContent = "";
  tops.forEach(([name, [k, p]]) => {
    const agrees = k === r.label;
    const d = el("details", "finding");
    const s = el("summary");
    s.append(el("h4", "", VIEW_NAMES[name]), el("span", "tag " + (agrees ? "agree" : "differ"), agrees ? "Agrees" : "Differs"), el("span", "chev"));
    s.append(el("span", "says", `Leans ${STAMPS[k].word.toLowerCase()} at ${pctText(p)}%. ${VIEW_HINT[name]}`));
    const body = el("div", "body"); const b = el("div"); body.append(b);
    bars(b, r.per_view[name]);
    d.append(s, body); f.append(d);
  });

  $("v-note").textContent = (r.in_training_data
    ? "This tweet was in the training data, so this score is optimistic. "
    : "") + "The model sorts tweets into rumour categories from wording and spread. It does not check facts against sources.";

  const sp = $("spread"); sp.textContent = "";
  if (r.spread) {
    const x = r.spread, dl = el("dl", "facts");
    [["Shares", x.nodes.toLocaleString()], ["Unique accounts", x.unique_users.toLocaleString()],
     ["First share after", fmtMin(x.first_delay_min)], ["Median delay", fmtMin(x.median_delay_min)],
     ["Within 5 min", pct(x.within_5min) + "%"], ["Within 1 hour", pct(x.within_60min) + "%"],
     ["Within 1 day", pct(x.within_1day) + "%"]].forEach(([k, v]) => {
      const row = el("div"); row.append(el("dt", "", k), el("dd", "", v)); dl.append(row);
    });
    sp.append(dl);
  } else {
    sp.append(el("p", "muted", "No spread data was given, so only the text view was used."));
  }
  bars($("bars"), r.probabilities, getComputedStyle(stamp).getPropertyValue("--c"));
}

/* ---------- examples ---------- */
async function loadSamples() {
  const ul = $("samples"); ul.textContent = "";
  ul.append(el("li", "muted", "Loading examples…"));
  try {
    const items = await api(`/api/samples?dataset=${$("ex-dataset").value}&n=6`);
    ul.textContent = "";
    items.forEach((s) => {
      const li = el("li"); const b = el("button"); b.type = "button";
      b.append(el("span", "t", s.text), el("span", "k", "filed as " + (STAMPS[s.true_label] ? STAMPS[s.true_label].word.toLowerCase() : s.true_label)));
      b.onclick = () => { $("dataset").value = $("ex-dataset").value; runCheck({ dataset: $("ex-dataset").value, sample_id: s.id }, s.text, true); };
      li.append(b); ul.append(li);
    });
  } catch (e) { ul.textContent = ""; ul.append(el("li", "error", e.message)); }
}

/* ---------- wiring ---------- */
$("form").addEventListener("submit", (e) => {
  e.preventDefault();
  const text = $("text").value.trim();
  if (!text) { showError("Enter the tweet text to check."); $("text").focus(); return; }
  let nodes;
  try { nodes = parseProp($("prop").value); } catch (err) { showError(err.message); $("prop").focus(); return; }
  showError(""); $("go").disabled = true;
  runCheck({ dataset: $("dataset").value, text, propagation: nodes.length ? nodes : null }, text, nodes.length > 0);
});
$("again").onclick = () => { show("intake"); $("text").focus(); };
document.querySelectorAll(".navbtn").forEach((b) => b.onclick = () => {
  show(b.dataset.view);
  if (b.dataset.view === "examples") loadSamples();
});
$("shuffle").onclick = loadSamples;
$("ex-dataset").onchange = loadSamples;
show("intake");
