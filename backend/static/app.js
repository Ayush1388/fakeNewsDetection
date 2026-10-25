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

/* ---------- model results ---------- */
const CLASS_LABEL = { "non-rumor": "Not a rumour", true: "Verified", false: "False", unverified: "Unverified" };
const PROTO_TITLE = { random: "Random split", "story0.5": "Unseen stories", "story0.3": "Unseen stories, strict", temporal: "Past to future" };
const f1 = (n) => n.toFixed(1);
const SHORT = { "non-rumor": "Not rumour", true: "Verified", false: "False", unverified: "Unverified" };
const PROTO_ORDER = ["random", "story0.5", "story0.3", "temporal"];

function table(headers, rows) {
  const wrap = el("div", "tablewrap"); const t = el("table", "t");
  const thead = el("thead"); const hr = el("tr");
  headers.forEach((h) => hr.append(el("th", "", h))); thead.append(hr); t.append(thead);
  const tb = el("tbody");
  rows.forEach((r) => { const tr = el("tr"); r.forEach((cell) => tr.append(cell)); tb.append(tr); });
  t.append(tb); wrap.append(t); return wrap;
}
function statCell(s, best, bar) {
  const td = el("td", best ? "best" : "");
  td.append(document.createTextNode(f1(s.mean)));
  if (s.sd != null) td.append(el("span", "sd", " ±" + f1(s.sd)));
  if (bar) { const b = el("span", "accbar"); const i = el("i"); i.style.width = s.mean + "%"; b.append(i); td.append(b); }
  return td;
}
function block(title, sub) {
  const b = el("div", "res-block"); b.append(el("h3", "", title)); if (sub) b.append(el("p", "sub", sub)); return b;
}

function renderResults(R, key) {
  const body = $("res-body"); body.textContent = "";
  const ds = R.datasets[key];

  const m = block("What we are using: " + R.model.name, R.model.summary);
  const card = el("div", "mcard");
  R.model.views.forEach((v) => { const d = el("div"); d.append(el("b", "", v.name), el("span", "", v.detail)); card.append(d); });
  m.append(card, el("p", "meta-line", `Final step: ${R.model.meta}. Trains on ${R.model.trains_on}.`));
  body.append(m);

  const a = block("Accuracy by test protocol", "Average over 10 runs with different seeds, ± one standard deviation. Harder protocols test tweets about stories the model has not seen.");
  const rows = ds.protocols.map((p) => {
    const first = el("td"); first.textContent = p.title;
    const best = Math.max(p.views.text.mean, p.views.spreaders.mean, p.views.cascade.mean);
    return [first, statCell(p.accuracy, true, true), statCell(p.macro_f1, false, false),
      statCell(p.views.text, p.views.text.mean === best, false), statCell(p.views.spreaders, p.views.spreaders.mean === best, false),
      statCell(p.views.cascade, p.views.cascade.mean === best, false)];
  });
  a.append(table(["Protocol", "GE-Stack acc.", "Macro F1", "Text only", "Spreaders only", "Cascade only"], rows));
  body.append(a);

  const w = block("What happens when a view is removed", "Accuracy of the stack trained without one view. A bigger drop means that view matters more.");
  w.append(table(["Protocol", "All views", "No text", "No spreaders", "No cascade"], ds.protocols.map((p) => {
    const first = el("td"); first.textContent = p.title;
    return [first, statCell(p.accuracy, true), statCell(p.without.text), statCell(p.without.spreaders), statCell(p.without.cascade)];
  })));
  body.append(w);

  const rnd = ds.protocols.find((p) => p.key === "random");
  const two = el("div", "two");
  const pc = block("Per-class F1", "Random split. Higher is better.");
  R.classes.forEach((c) => {
    const v = rnd.per_class_f1[c]; const row = el("div", "bar top");
    const track = el("div", "track"); const fill = el("div", "fill"); fill.style.width = v + "%"; track.append(fill);
    row.append(el("span", "", CLASS_LABEL[c]), track, el("span", "v", f1(v)));
    pc.append(row);
  });
  const cmBlock = block("Confusion matrix", "Random split, all 10 runs added up. Rows are the true class, columns the prediction.");
  const cm = el("div", "cm"); cm.append(el("span"));
  R.classes.forEach((c) => cm.append(el("span", "h", SHORT[c])));
  const maxv = Math.max(...ds.confusion_random.flat());
  ds.confusion_random.forEach((row, i) => {
    cm.append(el("span", "rowh", SHORT[R.classes[i]]));
    row.forEach((v, j) => {
      const cell = el("span", "c", String(v));
      const a2 = (v / maxv) * (i === j ? 0.55 : 0.9);
      cell.style.background = `color-mix(in srgb, ${i === j ? "var(--green)" : "var(--vermilion)"} ${Math.round(a2 * 100)}%, transparent)`;
      cm.append(cell);
    });
  });
  cmBlock.append(cm);
  two.append(pc, cmBlock); body.append(two);

  const cmp = R.neural_comparison.filter((r) => r.dataset === key);
  if (cmp.length) {
    const n = block("Against fine-tuned neural models", "Same test tweets for every model. GE-Stack is the model behind the rulings above.");
    const byProto = {};
    cmp.forEach((r) => { (byProto[r.protocol] = byProto[r.protocol] || { g: r.gestack, n: r.n_test, models: {} }).models[r.model] = r; });
    const names = [...new Set(cmp.map((r) => r.model))];
    n.append(table(["Protocol", "GE-Stack", ...names, "Gain over best"], Object.entries(byProto).sort((x, y) => PROTO_ORDER.indexOf(x[0]) - PROTO_ORDER.indexOf(y[0])).map(([proto, d]) => {
      const first = el("td"); first.textContent = PROTO_TITLE[proto] || proto;
      const best = Math.max(...names.map((m2) => d.models[m2].neural));
      const gcell = el("td", "best", f1(d.g));
      const cells = names.map((m2) => el("td", "", f1(d.models[m2].neural)));
      return [first, gcell, ...cells, el("td", "", "+" + f1(d.g - best))];
    })));
    n.append(el("p", "meta-line", "Neural models overfit here: with about 1,000 short tweets they memorise the training set. Test sets have 224 tweets, so each result carries a few points of noise."));
    body.append(n);
  }

  body.append(el("p", "meta-line", "Caution: random-split scores are optimistic, because near-duplicate tweets sit on both sides of the split. The harder protocols are a fairer guide to new stories."));
}

async function loadResults() {
  try {
    const R = await api("/api/results");
    const tabs = $("res-tabs"); tabs.textContent = "";
    const keys = Object.keys(R.datasets);
    const select = (k) => {
      tabs.querySelectorAll("button").forEach((b) => b.setAttribute("aria-selected", String(b.dataset.k === k)));
      renderResults(R, k);
    };
    keys.forEach((k) => {
      const b = el("button", "", R.datasets[k].label); b.type = "button"; b.role = "tab"; b.dataset.k = k;
      b.onclick = () => select(k); tabs.append(b);
    });
    select(keys[0]);
  } catch (e) { $("res-body").textContent = ""; $("res-body").append(el("p", "error", e.message)); }
}
loadResults();
