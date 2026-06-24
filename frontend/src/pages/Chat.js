import React, { useState, useEffect, useRef } from "react";

// ─────────────────────────────────────────────────────────────
//  HealthRAG-IN — Chat Page (matches "Clinical Glass" identity)
//   - Reveal/boot screen plays on arrival (esp. from landing CTA)
//   - Frosted glass message surfaces over the clinical field
//   - Citation pills, source drawer, provider badge, latency
//   - Space Grotesk display · Inter body · JetBrains Mono data
// ─────────────────────────────────────────────────────────────

const API_URL =
  process.env.NODE_ENV === "production"
    ? "https://onorog-healthrag-in-api.hf.space"
    : "http://localhost:8000";

const SAMPLE_QUESTIONS = [
  "What is HbA1c and why does it matter?",
  "What diet does ICMR recommend for diabetics in India?",
  "What is the difference between Type 1 and Type 2 diabetes?",
  "How is diabetic ketoacidosis treated?",
];

const BOOT_LINES = [
  "Loading retrieval index",
  "Warming semantic search",
  "Connecting language model",
  "Ready",
];

export default function Chat() {
  const [booting, setBooting] = useState(true);
  const [bootStep, setBootStep] = useState(0);

  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [stats, setStats] = useState(null);
  const [openSources, setOpenSources] = useState({});

  const endRef = useRef(null);
  const inputRef = useRef(null);

  // boot sequence
  useEffect(() => {
    const stepMs = 520;
    const timers = BOOT_LINES.map((_, i) =>
      setTimeout(() => setBootStep(i), i * stepMs)
    );
    const done = setTimeout(() => {
      setBooting(false);
      inputRef.current && inputRef.current.focus();
    }, BOOT_LINES.length * stepMs + 240);
    return () => {
      timers.forEach(clearTimeout);
      clearTimeout(done);
    };
  }, []);

  // fetch corpus stats
  useEffect(() => {
    fetch(`${API_URL}/stats`)
      .then((r) => r.json())
      .then(setStats)
      .catch(() => setStats(null));
  }, []);

  // autoscroll
  useEffect(() => {
    endRef.current && endRef.current.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  const sendQuestion = async (q) => {
    const question = (q ?? input).trim();
    if (!question || loading) return;

    setMessages((m) => [...m, { role: "user", text: question }]);
    setInput("");
    setLoading(true);

    try {
      const res = await fetch(`${API_URL}/ask`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question }),
      });
      if (!res.ok) throw new Error(`Server returned ${res.status}`);
      const data = await res.json();
      setMessages((m) => [
        ...m,
        {
          role: "assistant",
          text: data.answer,
          sources: data.retrieved_chunks || [],
          provider: data.provider || "unknown",
          latency: data.total_time_sec,
          isRefusal: data.is_refusal,
          isEmergency: data.is_emergency,
        },
      ]);
    } catch (err) {
      setMessages((m) => [
        ...m,
        {
          role: "assistant",
          text: "Something went wrong reaching the assistant. The service may be waking up from sleep — please try again in a moment.",
          sources: [],
          provider: "error",
          isError: true,
        },
      ]);
    } finally {
      setLoading(false);
      inputRef.current && inputRef.current.focus();
    }
  };

  const onKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendQuestion();
    }
  };

  const toggleSources = (i) =>
    setOpenSources((s) => ({ ...s, [i]: !s[i] }));

  // render answer text with citation pills
  const renderAnswer = (text) => {
    const parts = String(text).split(/(\[\d+(?:\s*,\s*\d+)*\])/g);
    return parts.map((part, i) => {
      const m = part.match(/^\[(\d+(?:\s*,\s*\d+)*)\]$/);
      if (m) {
        return (
          <span className="cite" key={i}>
            {m[1].replace(/\s/g, "")}
          </span>
        );
      }
      return <span key={i}>{part}</span>;
    });
  };

  return (
    <div className="chat-page">
      <style>{CSS}</style>

      {/* ambient background */}
      <div className="bg-field" aria-hidden="true" />
      <div className="bg-grid" aria-hidden="true" />
      <div className="bg-glow bg-glow-1" aria-hidden="true" />
      <div className="bg-glow bg-glow-2" aria-hidden="true" />

      {/* ── BOOT / REVEAL SCREEN ── */}
      {booting && (
        <div className="boot">
          <div className="boot-mark">
            <svg viewBox="0 0 32 32" width="54" height="54">
              <path d="M13 4h6v9h9v6h-9v9h-6v-9H4v-6h9V4z" fill="url(#bg-cross)" />
              <defs>
                <linearGradient id="bg-cross" x1="0" y1="0" x2="32" y2="32">
                  <stop offset="0" stopColor="#5EEAD4" />
                  <stop offset="1" stopColor="#38BDF8" />
                </linearGradient>
              </defs>
            </svg>
            <span className="boot-ring" />
          </div>
          <div className="boot-name">
            HealthRAG<span className="boot-in">-IN</span>
          </div>
          <div className="boot-status">
            {BOOT_LINES.map((line, i) => (
              <div
                key={line}
                className={`boot-line ${i <= bootStep ? "on" : ""} ${
                  i === BOOT_LINES.length - 1 && i <= bootStep ? "ready" : ""
                }`}
              >
                <span className="boot-tick">
                  {i < bootStep || (i === BOOT_LINES.length - 1 && i <= bootStep)
                    ? "✓"
                    : "•"}
                </span>
                {line}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* ── HEADER ── */}
      <header className="chat-head">
        <a className="chat-brand" href="/">
          <span className="brand-mark" aria-hidden="true">
            <svg viewBox="0 0 32 32" width="22" height="22">
              <path d="M13 4h6v9h9v6h-9v9h-6v-9H4v-6h9V4z" fill="url(#hcg)" />
              <defs>
                <linearGradient id="hcg" x1="0" y1="0" x2="32" y2="32">
                  <stop offset="0" stopColor="#5EEAD4" />
                  <stop offset="1" stopColor="#38BDF8" />
                </linearGradient>
              </defs>
            </svg>
          </span>
          <span className="brand-name">
            HealthRAG<span className="brand-in">-IN</span>
          </span>
        </a>
        {stats && (
          <div className="chat-stats">
            <span>{stats.documents} docs</span>
            <span className="dot">·</span>
            <span>{(stats.chunks || 0).toLocaleString()} passages</span>
            <span className="dot">·</span>
            <span>{(stats.sources || []).join(" · ")}</span>
          </div>
        )}
        <a className="chat-back" href="/">
          ← Home
        </a>
      </header>

      {/* ── CONVERSATION ── */}
      <main className="conv">
        {messages.length === 0 && !loading && (
          <div className="empty">
            <h1 className="empty-title">
              Ask HealthRAG<span className="brand-in">-IN</span>
            </h1>
            <p className="empty-sub">
              A grounded medical assistant for diabetes. Every answer is cited to
              WHO, PubMed, or ICMR sources.
            </p>
            <div className="suggests">
              {SAMPLE_QUESTIONS.map((q) => (
                <button
                  key={q}
                  className="suggest glass"
                  onClick={() => sendQuestion(q)}
                >
                  {q}
                </button>
              ))}
            </div>
          </div>
        )}

        {messages.map((msg, i) =>
          msg.role === "user" ? (
            <div className="row user" key={i}>
              <div className="bubble-user">{msg.text}</div>
            </div>
          ) : (
            <div className="row assistant" key={i}>
              <div className={`bubble-assistant glass ${msg.isError ? "err" : ""}`}>
                {msg.isEmergency && (
                  <div className="emergency">
                    ⚠ If this is a medical emergency, call your local emergency
                    number or go to the nearest hospital immediately.
                  </div>
                )}
                <div className="answer-text">{renderAnswer(msg.text)}</div>

                {!msg.isError && (
                  <div className="meta-row">
                    <span
                      className={`provider-badge ${
                        msg.isRefusal ? "refused" : ""
                      }`}
                    >
                      {msg.isRefusal ? "REFUSED" : (msg.provider || "").toUpperCase()}
                    </span>
                    {typeof msg.latency === "number" && (
                      <span className="latency">{msg.latency.toFixed(1)}s</span>
                    )}
                    {msg.sources && msg.sources.length > 0 && (
                      <button
                        className="src-toggle"
                        onClick={() => toggleSources(i)}
                      >
                        {openSources[i] ? "Hide" : "Show"} sources (
                        {msg.sources.length})
                      </button>
                    )}
                  </div>
                )}

                {openSources[i] && msg.sources && (
                  <div className="src-drawer">
                    {msg.sources.map((s, si) => (
                      <a
                        className="src-item"
                        key={si}
                        href={s.url}
                        target="_blank"
                        rel="noreferrer"
                      >
                        <span className="src-num">{si + 1}</span>
                        <span className="src-meta">
                          <span className="src-source">{s.source}</span>
                          <span className="src-title">{s.title}</span>
                        </span>
                      </a>
                    ))}
                  </div>
                )}
              </div>
            </div>
          )
        )}

        {loading && (
          <div className="row assistant">
            <div className="bubble-assistant glass thinking">
              <span className="think-dot" />
              <span className="think-dot" />
              <span className="think-dot" />
              <span className="think-label">retrieving &amp; reasoning…</span>
            </div>
          </div>
        )}

        <div ref={endRef} />
      </main>

      {/* ── COMPOSER ── */}
      <footer className="composer-wrap">
        <div className="composer glass">
          <textarea
            ref={inputRef}
            className="composer-input"
            placeholder="Ask about diabetes — symptoms, HbA1c, diet, treatment…"
            value={input}
            rows={1}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={onKeyDown}
            disabled={loading}
          />
          <button
            className="composer-send"
            onClick={() => sendQuestion()}
            disabled={loading || !input.trim()}
            aria-label="Send question"
          >
            <svg viewBox="0 0 24 24" width="20" height="20" fill="none">
              <path
                d="M5 12h14M13 6l6 6-6 6"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
            </svg>
          </button>
        </div>
        <p className="composer-note">
          Educational tool · not medical advice · consult a qualified physician
        </p>
      </footer>
    </div>
  );
}

const CSS = `
:root{
  --bg-0:#06141B;
  --teal:#5EEAD4;
  --sky:#38BDF8;
  --ink:#E6F1F0;
  --ink-soft:#9FB7B5;
  --ink-faint:#6B8584;
  --glass:rgba(255,255,255,0.055);
  --glass-strong:rgba(255,255,255,0.09);
  --glass-border:rgba(148,222,216,0.18);
  --glass-border-soft:rgba(255,255,255,0.10);
  --radius:18px;
  --maxw:840px;
  --font-display:'Space Grotesk','Segoe UI',system-ui,sans-serif;
  --font-body:'Inter','Segoe UI',system-ui,sans-serif;
  --font-mono:'JetBrains Mono','SFMono-Regular',monospace;
}
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@500&display=swap');

*{box-sizing:border-box;margin:0;padding:0;}
html{scroll-behavior:smooth;}
body{background:var(--bg-0);}

.chat-page{
  position:relative;min-height:100vh;
  display:flex;flex-direction:column;
  font-family:var(--font-body);color:var(--ink);overflow-x:hidden;
}

/* background */
.bg-field{position:fixed;inset:0;z-index:-4;
  background:
    radial-gradient(1200px 800px at 70% -10%, #0E3A40 0%, transparent 55%),
    radial-gradient(1000px 700px at 10% 20%, #0B2E33 0%, transparent 50%),
    linear-gradient(180deg, #06141B 0%, #071A20 50%, #06141B 100%);}
.bg-grid{position:fixed;inset:0;z-index:-3;
  background-image:
    linear-gradient(rgba(94,234,212,0.035) 1px, transparent 1px),
    linear-gradient(90deg, rgba(94,234,212,0.035) 1px, transparent 1px);
  background-size:60px 60px;
  mask-image:radial-gradient(1000px 700px at 50% 0%, #000 0%, transparent 75%);}
.bg-glow{position:fixed;z-index:-2;border-radius:50%;filter:blur(120px);opacity:0.45;}
.bg-glow-1{width:440px;height:440px;background:rgba(56,189,248,0.20);top:-120px;right:-80px;}
.bg-glow-2{width:400px;height:400px;background:rgba(94,234,212,0.16);bottom:0;left:-100px;}

.glass{
  background:var(--glass);
  backdrop-filter:blur(20px) saturate(140%);
  -webkit-backdrop-filter:blur(20px) saturate(140%);
  border:1px solid var(--glass-border-soft);
  border-radius:var(--radius);
  box-shadow:0 8px 40px rgba(0,0,0,0.28), inset 0 1px 0 rgba(255,255,255,0.06);
}
.brand-in{color:var(--teal);}

/* ── boot ── */
.boot{
  position:fixed;inset:0;z-index:100;
  display:flex;flex-direction:column;align-items:center;justify-content:center;
  background:radial-gradient(900px 600px at 50% 40%, #0A2A30 0%, #06141B 70%);
  animation:bootOut .5s ease forwards;animation-delay:2.3s;
}
@keyframes bootOut{to{opacity:0;visibility:hidden;}}
.boot-mark{position:relative;display:flex;align-items:center;justify-content:center;margin-bottom:26px;
  filter:drop-shadow(0 0 22px rgba(94,234,212,0.5));animation:markIn .8s cubic-bezier(.2,.8,.2,1);}
@keyframes markIn{from{opacity:0;transform:scale(.6) rotate(-25deg);}to{opacity:1;transform:scale(1) rotate(0);}}
.boot-ring{position:absolute;width:96px;height:96px;border-radius:50%;
  border:1.5px solid rgba(94,234,212,0.25);border-top-color:var(--teal);
  animation:spin 1.1s linear infinite;}
@keyframes spin{to{transform:rotate(360deg);}}
.boot-name{font-family:var(--font-display);font-weight:700;font-size:34px;letter-spacing:-0.02em;
  color:var(--ink);margin-bottom:30px;animation:fadeUp .7s ease .2s both;}
.boot-in{background:linear-gradient(135deg,var(--teal),var(--sky));
  -webkit-background-clip:text;background-clip:text;-webkit-text-fill-color:transparent;}
.boot-status{display:flex;flex-direction:column;gap:9px;min-width:230px;}
.boot-line{display:flex;align-items:center;gap:10px;
  font-family:var(--font-mono);font-size:13px;color:var(--ink-faint);
  opacity:0.35;transition:opacity .4s, color .4s;}
.boot-line.on{opacity:1;color:var(--ink-soft);}
.boot-line.ready{color:var(--teal);}
.boot-tick{display:inline-flex;width:16px;justify-content:center;color:var(--teal);}
@keyframes fadeUp{from{opacity:0;transform:translateY(12px);}to{opacity:1;transform:translateY(0);}}

/* ── header ── */
.chat-head{
  position:sticky;top:0;z-index:20;
  display:flex;align-items:center;justify-content:space-between;gap:16px;
  padding:14px 22px;margin:14px auto 0;width:calc(100% - 28px);max-width:var(--maxw);
  background:rgba(8,24,28,0.55);
  backdrop-filter:blur(18px) saturate(140%);-webkit-backdrop-filter:blur(18px) saturate(140%);
  border:1px solid var(--glass-border-soft);border-radius:14px;
}
.chat-brand{display:flex;align-items:center;gap:10px;text-decoration:none;}
.brand-mark{display:flex;filter:drop-shadow(0 0 8px rgba(94,234,212,0.4));}
.brand-name{font-family:var(--font-display);font-weight:700;font-size:18px;color:var(--ink);letter-spacing:-0.01em;}
.chat-stats{display:flex;align-items:center;gap:8px;font-family:var(--font-mono);font-size:11.5px;color:var(--ink-faint);}
.chat-stats .dot{opacity:0.5;}
.chat-back{color:var(--ink-soft);text-decoration:none;font-size:13.5px;font-weight:500;transition:color .2s;}
.chat-back:hover{color:var(--teal);}

/* ── conversation ── */
.conv{
  flex:1;width:100%;max-width:var(--maxw);margin:0 auto;
  padding:28px 22px 140px;display:flex;flex-direction:column;gap:20px;
}

/* empty state */
.empty{text-align:center;margin-top:8vh;animation:fadeUp .6s ease;}
.empty-title{font-family:var(--font-display);font-weight:700;font-size:clamp(32px,6vw,52px);
  letter-spacing:-0.03em;color:var(--ink);}
.empty-sub{max-width:480px;margin:16px auto 0;font-size:16px;line-height:1.6;color:var(--ink-soft);}
.suggests{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:34px;text-align:left;}
.suggest{padding:16px 18px;font-family:var(--font-body);font-size:14.5px;color:var(--ink);
  cursor:pointer;transition:transform .2s, border-color .2s, background .2s;line-height:1.4;}
.suggest:hover{transform:translateY(-2px);border-color:var(--glass-border);background:var(--glass-strong);}

/* rows */
.row{display:flex;animation:msgIn .4s ease;}
.row.user{justify-content:flex-end;}
.row.assistant{justify-content:flex-start;}
@keyframes msgIn{from{opacity:0;transform:translateY(10px);}to{opacity:1;transform:translateY(0);}}

.bubble-user{
  max-width:78%;padding:14px 18px;border-radius:16px 16px 4px 16px;
  font-size:15px;line-height:1.5;color:#04181A;font-weight:500;
  background:linear-gradient(135deg,var(--teal),var(--sky));
  box-shadow:0 6px 20px rgba(56,189,248,0.25);
}
.bubble-assistant{
  max-width:88%;padding:20px 22px;border-radius:16px 16px 16px 4px;
}
.bubble-assistant.err{border-color:rgba(248,113,113,0.4);}
.answer-text{font-size:15px;line-height:1.72;color:var(--ink);white-space:pre-wrap;}

.emergency{
  font-size:13.5px;line-height:1.5;color:#FCA5A5;font-weight:600;
  padding:12px 14px;margin-bottom:14px;border-radius:10px;
  background:rgba(248,113,113,0.1);border:1px solid rgba(248,113,113,0.3);
}

.cite{
  display:inline-flex;align-items:center;justify-content:center;
  font-family:var(--font-mono);font-size:10px;font-weight:600;
  min-width:16px;height:16px;padding:0 4px;margin:0 2px;
  border-radius:5px;color:var(--teal);
  background:rgba(94,234,212,0.14);border:1px solid rgba(94,234,212,0.3);
  vertical-align:super;
}

.meta-row{display:flex;align-items:center;gap:12px;margin-top:16px;
  padding-top:14px;border-top:1px solid var(--glass-border-soft);flex-wrap:wrap;}
.provider-badge{font-family:var(--font-mono);font-size:10.5px;letter-spacing:0.05em;
  color:var(--teal);padding:4px 10px;border-radius:7px;
  background:rgba(94,234,212,0.1);border:1px solid rgba(94,234,212,0.25);}
.provider-badge.refused{color:#FCD34D;background:rgba(252,211,77,0.1);border-color:rgba(252,211,77,0.3);}
.latency{font-family:var(--font-mono);font-size:11px;color:var(--ink-faint);}
.src-toggle{margin-left:auto;font-family:var(--font-body);font-size:12.5px;font-weight:500;
  color:var(--ink-soft);background:none;border:none;cursor:pointer;transition:color .2s;}
.src-toggle:hover{color:var(--teal);}

.src-drawer{margin-top:14px;display:flex;flex-direction:column;gap:8px;}
.src-item{display:flex;gap:12px;padding:12px 14px;border-radius:10px;text-decoration:none;
  background:rgba(255,255,255,0.04);border:1px solid var(--glass-border-soft);transition:border-color .2s, background .2s;}
.src-item:hover{border-color:var(--glass-border);background:rgba(255,255,255,0.06);}
.src-num{flex-shrink:0;width:22px;height:22px;border-radius:6px;display:flex;align-items:center;justify-content:center;
  font-family:var(--font-mono);font-size:11px;color:var(--teal);
  background:rgba(94,234,212,0.12);border:1px solid rgba(94,234,212,0.25);}
.src-meta{display:flex;flex-direction:column;gap:3px;min-width:0;}
.src-source{font-family:var(--font-mono);font-size:10.5px;letter-spacing:0.05em;color:var(--ink-faint);text-transform:uppercase;}
.src-title{font-size:13px;color:var(--ink-soft);line-height:1.4;}

/* thinking */
.thinking{display:flex;align-items:center;gap:7px;padding:18px 22px;}
.think-dot{width:8px;height:8px;border-radius:50%;background:var(--teal);opacity:0.4;animation:think 1.2s infinite;}
.think-dot:nth-child(2){animation-delay:.2s;}
.think-dot:nth-child(3){animation-delay:.4s;}
.think-label{margin-left:8px;font-family:var(--font-mono);font-size:12px;color:var(--ink-faint);}
@keyframes think{0%,60%,100%{opacity:0.3;transform:translateY(0);}30%{opacity:1;transform:translateY(-4px);}}

/* composer */
.composer-wrap{
  position:fixed;bottom:0;left:0;right:0;z-index:20;
  padding:16px 22px 18px;
  display:flex;flex-direction:column;align-items:center;gap:8px;
  background:linear-gradient(180deg, transparent, var(--bg-0) 45%);
}
.composer{
  width:100%;max-width:var(--maxw);display:flex;align-items:flex-end;gap:10px;
  padding:10px 10px 10px 18px;border-radius:16px;
}
.composer-input{
  flex:1;background:none;border:none;outline:none;resize:none;
  font-family:var(--font-body);font-size:15px;color:var(--ink);line-height:1.5;
  max-height:140px;padding:8px 0;
}
.composer-input::placeholder{color:var(--ink-faint);}
.composer-send{
  flex-shrink:0;width:44px;height:44px;border-radius:12px;border:none;cursor:pointer;
  display:flex;align-items:center;justify-content:center;color:#04181A;
  background:linear-gradient(135deg,var(--teal),var(--sky));
  transition:transform .2s, box-shadow .2s, opacity .2s;
}
.composer-send:hover:not(:disabled){transform:translateY(-2px);box-shadow:0 8px 22px rgba(56,189,248,0.4);}
.composer-send:disabled{opacity:0.4;cursor:not-allowed;}
.composer-note{font-family:var(--font-mono);font-size:11px;color:var(--ink-faint);}

@media(max-width:640px){
  .suggests{grid-template-columns:1fr;}
  .chat-stats{display:none;}
  .bubble-assistant{max-width:94%;}
  .bubble-user{max-width:88%;}
}
@media(prefers-reduced-motion:reduce){
  *{animation-duration:.01ms!important;}
  .boot{animation-delay:1.2s;}
}
`;
