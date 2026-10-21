const $ = (id) => document.getElementById(id);
const pct = (p) => (p * 100).toFixed(1) + "%";

function show(el, on) { el.hidden = !on; }
function showError(msg) { $("error").textContent = msg || ""; show($("error"), !!msg); }

function bars(container, probs) {
  container.textContent = "";
  Object.entries(probs).sort((a, b) => b[1] - a[1]).forEach(([k, v]) => {
    const row = document.createElement("div");
    row.className = "bar";
    const name = document.createElement("span"); name.textContent = k;
    const track = document.createElement("div"); track.className = "track";
    const fill = document.createElement("div"); fill.className = "fill"; fill.style.width = pct(v);
    track.append(fill);
    const val = document.createElement("span"); val.textContent = pct(v);
    row.append(name, track, val);
    container.append(row);
  });
}

function parseProp(raw) {
  const nodes = [];
  for (const [i, line] of raw.split("\n").entries()) {
    if (!line.trim()) continue;
    const [user, delay] = line.split(/[,\s]+/).filter(Boolean);
    const d = Number(delay);
    if (!user || delay === undefined || !Number.isFinite(d) || d < 0)
      throw new Error(`Propagation line ${i + 1}: expected "user_id, delay_minutes".`);
    nodes.push({ user, delay_minutes: d });
  }
  return nodes;
}

async function api(path, opts) {
  const res = await fetch(path, opts);
  if (!res.ok) {
    let detail = res.statusText;
    try { const j = await res.json(); detail = typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail); } catch {}
    throw new Error(detail);
  }
  return res.json();
}

function render(r) {
  $("label").textContent = r.label;
  $("conf").textContent = pct(r.confidence) + " confidence";
  const used = r.views_used.join(" + ");
  $("note").textContent = `Views used: ${used}` +
    (r.propagation_nodes ? ` (${r.propagation_nodes} spreaders)` : " — no propagation given, text only") +
    (r.in_training_data ? ". This tweet was in the training data." : "");
  bars($("bars"), r.probabilities);
  $("views").textContent = "";
  Object.entries(r.per_view).forEach(([name, probs]) => {
    const wrap = document.createElement("div"); wrap.className = "view";
    const t = document.createElement("b"); t.textContent = name;
    const b = document.createElement("div");
    wrap.append(t, b); $("views").append(wrap);
    bars(b, probs);
  });
  show($("result"), true);
}

async function analyse(body) {
  showError("");
  $("go").disabled = true;
  try { render(await api("/api/predict", {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
  })); }
  catch (e) { showError(e.message); }
  finally { $("go").disabled = false; }
}

$("go").onclick = () => {
  const text = $("text").value.trim();
  if (!text) return showError("Enter some tweet text.");
  let nodes;
  try { nodes = parseProp($("prop").value); } catch (e) { return showError(e.message); }
  analyse({ dataset: $("dataset").value, text, propagation: nodes.length ? nodes : null });
};

$("clear").onclick = () => { $("text").value = ""; $("prop").value = ""; showError(""); show($("result"), false); };

async function loadSamples() {
  const ul = $("samples"); ul.textContent = "Loading…";
  try {
    const items = await api(`/api/samples?dataset=${$("dataset").value}&n=6`);
    ul.textContent = "";
    items.forEach((s) => {
      const li = document.createElement("li"); li.tabIndex = 0;
      li.textContent = s.text;
      const small = document.createElement("small"); small.textContent = `true label: ${s.true_label}`;
      li.append(small);
      const pick = () => { $("text").value = s.text; $("prop").value = "";
        analyse({ dataset: $("dataset").value, sample_id: s.id }); };
      li.onclick = pick; li.onkeydown = (e) => { if (e.key === "Enter") pick(); };
      ul.append(li);
    });
  } catch (e) { ul.textContent = ""; showError(e.message); }
}

$("reload").onclick = loadSamples;
$("dataset").onchange = loadSamples;
loadSamples();
