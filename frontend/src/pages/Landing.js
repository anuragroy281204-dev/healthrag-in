import React, { useState, useEffect, useRef } from "react";

// ─────────────────────────────────────────────────────────────
//  HealthRAG-IN — Landing Page (industry-grade redesign)
//  Design language: "Clinical Glass"
//   - Deep teal-navy clinical gradient field
//   - Frosted glass panels (glassmorphism) as the primary surface
//   - Display: Space Grotesk (geometric, confident, NOT italic)
//   - Body: Inter (clean, legible, professional)
//   - Mono: JetBrains Mono (citations, data, code accents)
//   - HealthRAG-IN is the unmistakable hero — large, centered, owned
// ─────────────────────────────────────────────────────────────


function useCountUp(target, duration = 1600, start = false) {
  const [value, setValue] = useState(0);
  useEffect(() => {
    if (!start) return;
    let raf;
    const t0 = performance.now();
    const tick = (now) => {
      const p = Math.min((now - t0) / duration, 1);
      const eased = 1 - Math.pow(1 - p, 3);
      setValue(Math.round(eased * target));
      if (p < 1) raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [target, duration, start]);
  return value;
}

function useInView(threshold = 0.25) {
  const ref = useRef(null);
  const [inView, setInView] = useState(false);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const obs = new IntersectionObserver(
      ([entry]) => entry.isIntersecting && setInView(true),
      { threshold }
    );
    obs.observe(el);
    return () => obs.disconnect();
  }, [threshold]);
  return [ref, inView];
}

export default function Landing() {
  const [statsRef, statsInView] = useInView(0.4);
  const docs = useCountUp(490, 1500, statsInView);
  const chunks = useCountUp(1247, 1700, statsInView);
  const sources = useCountUp(3, 1000, statsInView);

  const [howRef, howInView] = useInView(0.2);
  const [srcRef, srcInView] = useInView(0.2);
  const [whyRef, whyInView] = useInView(0.2);

  const goToChat = () => {
    window.location.href = "/chat";
  };

  return (
    <div className="page">
      <style>{CSS}</style>

      {/* ambient background layers */}
      <div className="bg-field" aria-hidden="true" />
      <div className="bg-grid" aria-hidden="true" />
      <div className="bg-glow bg-glow-1" aria-hidden="true" />
      <div className="bg-glow bg-glow-2" aria-hidden="true" />

      {/* ── NAV ── */}
      <nav className="nav">
        <div className="nav-inner">
          <div className="brand">
            <span className="brand-mark" aria-hidden="true">
              <svg viewBox="0 0 32 32" width="26" height="26">
                <path
                  d="M13 4h6v9h9v6h-9v9h-6v-9H4v-6h9V4z"
                  fill="url(#cg)"
                />
                <defs>
                  <linearGradient id="cg" x1="0" y1="0" x2="32" y2="32">
                    <stop offset="0" stopColor="#5EEAD4" />
                    <stop offset="1" stopColor="#38BDF8" />
                  </linearGradient>
                </defs>
              </svg>
            </span>
            <span className="brand-name">
              HealthRAG<span className="brand-in">-IN</span>
            </span>
          </div>
          <div className="nav-links">
            <a href="#how">How it works</a>
            <a href="#sources">Sources</a>
            <a href="#why">Why grounded</a>
            <button className="nav-cta" onClick={goToChat}>
              Open the assistant
            </button>
          </div>
        </div>
      </nav>

      {/* ── HERO ── */}
      <header className="hero">
        <div className="hero-eyebrow">
          <span className="pulse-dot" /> Grounded medical Q&amp;A · Diabetes care
        </div>

        <h1 className="hero-title">
          HealthRAG<span className="hero-in">-IN</span>
        </h1>

        <p className="hero-sub">
          A medical question-answering assistant that answers only from real
          clinical sources — WHO, PubMed, and ICMR — and cites every claim.
          When the evidence isn&apos;t there, it says so instead of guessing.
        </p>

        <div className="hero-actions">
          <button className="btn-primary" onClick={goToChat}>
            Ask a question
            <svg viewBox="0 0 24 24" width="18" height="18" fill="none">
              <path
                d="M5 12h14M13 6l6 6-6 6"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
            </svg>
          </button>
          <a className="btn-ghost" href="#how">
            See how it works
          </a>
        </div>

        {/* glass demo card */}
        <div className="hero-demo glass">
          <div className="demo-head">
            <span className="demo-q-label">Question</span>
            <span className="demo-provider">live · cited</span>
          </div>
          <p className="demo-q">What is HbA1c and why does it matter?</p>
          <div className="demo-divider" />
          <p className="demo-a">
            HbA1c reflects average blood glucose over the preceding 8–12 weeks
            <span className="cite">1</span> and is used to diagnose and monitor
            diabetes<span className="cite">1</span>. ICMR recommends a target
            below 7% for most Indian adults<span className="cite">2</span>.
          </p>
          <div className="demo-sources">
            <span className="src-chip">WHO</span>
            <span className="src-chip">ICMR</span>
            <span className="src-chip">PubMed</span>
          </div>
        </div>
      </header>

      {/* ── STATS ── */}
      <section className="stats" ref={statsRef}>
        <div className="stat glass">
          <div className="stat-num">{docs}</div>
          <div className="stat-label">Source documents</div>
        </div>
        <div className="stat glass">
          <div className="stat-num">{chunks.toLocaleString()}</div>
          <div className="stat-label">Indexed passages</div>
        </div>
        <div className="stat glass">
          <div className="stat-num">{sources}</div>
          <div className="stat-label">Trusted authorities</div>
        </div>
        <div className="stat glass">
          <div className="stat-num">
            100<span className="stat-pct">%</span>
          </div>
          <div className="stat-label">Claims cited</div>
        </div>
      </section>

      {/* ── HOW IT WORKS ── */}
      <section className="section" id="how" ref={howRef}>
        <div className="section-head">
          <span className="eyebrow">The method</span>
          <h2 className="section-title">How an answer is built</h2>
          <p className="section-lead">
            Every response travels the same path. Nothing is invented along the
            way.
          </p>
        </div>

        <div className={`steps ${howInView ? "in" : ""}`}>
          {STEPS.map((s, i) => (
            <div className="step glass" key={s.title} style={{ "--d": `${i * 90}ms` }}>
              <div className="step-index">{String(i + 1).padStart(2, "0")}</div>
              <h3 className="step-title">{s.title}</h3>
              <p className="step-body">{s.body}</p>
            </div>
          ))}
        </div>
      </section>

      {/* ── SOURCES ── */}
      <section className="section" id="sources" ref={srcRef}>
        <div className="section-head">
          <span className="eyebrow">The knowledge base</span>
          <h2 className="section-title">Three authorities, one corpus</h2>
          <p className="section-lead">
            Chosen for breadth, research depth, and India-specific clinical
            relevance.
          </p>
        </div>

        <div className={`cards ${srcInView ? "in" : ""}`}>
          {SOURCES.map((s, i) => (
            <div className="card glass" key={s.name} style={{ "--d": `${i * 110}ms` }}>
              <div className="card-tag">{s.tag}</div>
              <h3 className="card-title">{s.name}</h3>
              <p className="card-body">{s.body}</p>
              <div className="card-count">{s.count}</div>
            </div>
          ))}
        </div>
      </section>

      {/* ── WHY GROUNDED ── */}
      <section className="section" id="why" ref={whyRef}>
        <div className="section-head">
          <span className="eyebrow">The principle</span>
          <h2 className="section-title">Why grounding matters in medicine</h2>
          <p className="section-lead">
            A confident wrong answer is worse than no answer. The system is built
            around that idea.
          </p>
        </div>

        <div className={`why-grid ${whyInView ? "in" : ""}`}>
          {WHY.map((w, i) => (
            <div className="why-item glass" key={w.title} style={{ "--d": `${i * 80}ms` }}>
              <div className="why-icon" aria-hidden="true">
                {w.icon}
              </div>
              <h3 className="why-title">{w.title}</h3>
              <p className="why-body">{w.body}</p>
            </div>
          ))}
        </div>
      </section>

      {/* ── CTA ── */}
      <section className="cta-band">
        <div className="cta-card glass">
          <h2 className="cta-title">Ask it something about diabetes.</h2>
          <p className="cta-sub">
            Watch it cite its sources — or decline when it shouldn&apos;t answer.
          </p>
          <button className="btn-primary lg" onClick={goToChat}>
            Open HealthRAG-IN
            <svg viewBox="0 0 24 24" width="18" height="18" fill="none">
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
      </section>

      {/* ── FOOTER ── */}
      <footer className="footer">
        <div className="footer-inner">
          <div className="footer-col footer-brand-col">
            <div className="brand">
              <span className="brand-mark" aria-hidden="true">
                <svg viewBox="0 0 32 32" width="22" height="22">
                  <path d="M13 4h6v9h9v6h-9v9h-6v-9H4v-6h9V4z" fill="url(#cg2)" />
                  <defs>
                    <linearGradient id="cg2" x1="0" y1="0" x2="32" y2="32">
                      <stop offset="0" stopColor="#5EEAD4" />
                      <stop offset="1" stopColor="#38BDF8" />
                    </linearGradient>
                  </defs>
                </svg>
              </span>
              <span className="brand-name sm">
                HealthRAG<span className="brand-in">-IN</span>
              </span>
            </div>
            <p className="footer-tagline">
              Grounded medical question answering for diabetes care, built on
              retrieval-augmented generation.
            </p>
          </div>

          <div className="footer-col">
            <h4 className="footer-h">Explore</h4>
            <a href="#how">How it works</a>
            <a href="#sources">Sources</a>
            <a href="#why">Why grounded</a>
            <a href="/chat">Open assistant</a>
          </div>

          <div className="footer-col">
            <h4 className="footer-h">Project</h4>
            <a
              href="https://github.com/anuragroy281204-dev/healthrag-in"
              target="_blank"
              rel="noreferrer"
            >
              Source code
            </a>
            <a
              href="https://onorog-healthrag-in-api.hf.space/health"
              target="_blank"
              rel="noreferrer"
            >
              API status
            </a>
            <span className="footer-static">Built by Anurag Roy</span>
            <span className="footer-static">B.Tech CSE (Data Science)</span>
          </div>

          <div className="footer-col">
            <h4 className="footer-h">Contact</h4>
            <a href="mailto:royanurag281204@gmail.com">
              royanurag281204@gmail.com
            </a>
            <a
              href="https://github.com/anuragroy281204-dev"
              target="_blank"
              rel="noreferrer"
            >
              github.com/anuragroy281204-dev
            </a>
          </div>
        </div>

        {/* subtle limitations notice */}
        <div className="footer-limits">
          <p>
            <strong>Important:</strong> HealthRAG-IN is an educational research
            tool, not a medical device. Its knowledge is limited to a fixed
            corpus of WHO, PubMed, and ICMR documents focused on diabetes, so it
            may not reflect the latest guidance, may lack coverage on adjacent
            topics, and can occasionally retrieve imperfectly. It does not
            provide diagnosis or personalised treatment advice. Always consult a
            qualified physician for medical decisions.
          </p>
        </div>

        <div className="footer-base">
          <span>© 2026 HealthRAG-IN · Anurag Roy</span>
          <span className="footer-base-note">
            Not medical advice · For educational use only
          </span>
        </div>
      </footer>
    </div>
  );
}

// ── content ──
const STEPS = [
  {
    title: "Retrieve",
    body: "Your question is matched against 1,247 indexed passages using both semantic meaning and exact keywords.",
  },
  {
    title: "Fuse",
    body: "The two search methods are merged with Reciprocal Rank Fusion to surface the five most relevant passages.",
  },
  {
    title: "Generate",
    body: "A language model writes the answer using only those passages — never its own training memory.",
  },
  {
    title: "Cite",
    body: "Every factual claim is tagged to its source, and out-of-scope questions are refused outright.",
  },
];

const SOURCES = [
  {
    tag: "Global",
    name: "WHO",
    body: "World Health Organization fact sheets on diabetes and its common comorbidities.",
    count: "4 documents",
  },
  {
    tag: "Research",
    name: "PubMed",
    body: "Peer-reviewed research abstracts from 2020 onward for depth and recency.",
    count: "480 abstracts",
  },
  {
    tag: "India",
    name: "ICMR",
    body: "Indian Council of Medical Research clinical guidelines and dietary guidance.",
    count: "6 guidelines",
  },
];

const WHY = [
  {
    icon: "❝",
    title: "Every claim is cited",
    body: "Answers carry inline markers linking each statement to a real document you can open and verify.",
  },
  {
    icon: "⛔",
    title: "It refuses to guess",
    body: "If the corpus can't support an answer, the system declines rather than fabricating one.",
  },
  {
    icon: "⚕",
    title: "Redirects personal advice",
    body: "Questions needing a doctor's judgement are pointed to a qualified physician, not answered.",
  },
  {
    icon: "🇮🇳",
    title: "India-aware",
    body: "ICMR guidelines bring local clinical targets and dietary context most models lack.",
  },
  {
    icon: "⚖",
    title: "Independently evaluated",
    body: "A separate model judges answer quality across five metrics to reduce self-bias.",
  },
  {
    icon: "↻",
    title: "Resilient by design",
    body: "A multi-provider fallback chain keeps the assistant online when any single model fails.",
  },
];

const CSS = `
:root{
  --bg-0:#06141B;
  --bg-1:#0A2A2F;
  --teal:#5EEAD4;
  --sky:#38BDF8;
  --ink:#E6F1F0;
  --ink-soft:#9FB7B5;
  --ink-faint:#6B8584;
  --glass:rgba(255,255,255,0.055);
  --glass-strong:rgba(255,255,255,0.09);
  --glass-border:rgba(148,222,216,0.18);
  --glass-border-soft:rgba(255,255,255,0.10);
  --radius:20px;
  --maxw:1180px;
  --font-display:'Space Grotesk','Segoe UI',system-ui,sans-serif;
  --font-body:'Inter','Segoe UI',system-ui,sans-serif;
  --font-mono:'JetBrains Mono','SFMono-Regular',monospace;
}

@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@500&display=swap');

*{box-sizing:border-box;margin:0;padding:0;}
html{scroll-behavior:smooth;}
body{background:var(--bg-0);}

.page{
  position:relative;
  min-height:100vh;
  font-family:var(--font-body);
  color:var(--ink);
  overflow-x:hidden;
}

/* ── background ── */
.bg-field{
  position:fixed;inset:0;z-index:-4;
  background:
    radial-gradient(1200px 800px at 70% -10%, #0E3A40 0%, transparent 55%),
    radial-gradient(1000px 700px at 10% 20%, #0B2E33 0%, transparent 50%),
    linear-gradient(180deg, #06141B 0%, #071A20 50%, #06141B 100%);
}
.bg-grid{
  position:fixed;inset:0;z-index:-3;
  background-image:
    linear-gradient(rgba(94,234,212,0.035) 1px, transparent 1px),
    linear-gradient(90deg, rgba(94,234,212,0.035) 1px, transparent 1px);
  background-size:60px 60px;
  mask-image:radial-gradient(1000px 700px at 50% 0%, #000 0%, transparent 75%);
}
.bg-glow{position:fixed;z-index:-2;border-radius:50%;filter:blur(120px);opacity:0.5;}
.bg-glow-1{width:480px;height:480px;background:rgba(56,189,248,0.22);top:-120px;right:-80px;animation:float1 18s ease-in-out infinite;}
.bg-glow-2{width:420px;height:420px;background:rgba(94,234,212,0.18);bottom:5%;left:-100px;animation:float2 22s ease-in-out infinite;}
@keyframes float1{0%,100%{transform:translate(0,0);}50%{transform:translate(-40px,40px);}}
@keyframes float2{0%,100%{transform:translate(0,0);}50%{transform:translate(40px,-30px);}}

/* ── glass primitive ── */
.glass{
  background:var(--glass);
  backdrop-filter:blur(20px) saturate(140%);
  -webkit-backdrop-filter:blur(20px) saturate(140%);
  border:1px solid var(--glass-border-soft);
  border-radius:var(--radius);
  box-shadow:0 8px 40px rgba(0,0,0,0.28), inset 0 1px 0 rgba(255,255,255,0.06);
}

/* ── nav ── */
.nav{position:sticky;top:0;z-index:50;padding:14px 24px;}
.nav-inner{
  max-width:var(--maxw);margin:0 auto;
  display:flex;align-items:center;justify-content:space-between;
  padding:12px 22px;
  background:rgba(8,24,28,0.55);
  backdrop-filter:blur(18px) saturate(140%);
  -webkit-backdrop-filter:blur(18px) saturate(140%);
  border:1px solid var(--glass-border-soft);
  border-radius:16px;
}
.brand{display:flex;align-items:center;gap:11px;}
.brand-mark{display:flex;filter:drop-shadow(0 0 10px rgba(94,234,212,0.4));}
.brand-name{
  font-family:var(--font-display);font-weight:700;font-size:20px;
  letter-spacing:-0.01em;color:var(--ink);
}
.brand-name.sm{font-size:18px;}
.brand-in{color:var(--teal);}
.nav-links{display:flex;align-items:center;gap:28px;}
.nav-links a{
  color:var(--ink-soft);text-decoration:none;font-size:14.5px;font-weight:500;
  transition:color .2s;
}
.nav-links a:hover{color:var(--ink);}
.nav-cta{
  font-family:var(--font-body);font-weight:600;font-size:14px;
  color:#04181A;background:linear-gradient(135deg,var(--teal),var(--sky));
  border:none;border-radius:10px;padding:10px 18px;cursor:pointer;
  transition:transform .2s, box-shadow .2s;
}
.nav-cta:hover{transform:translateY(-1px);box-shadow:0 8px 24px rgba(56,189,248,0.35);}

/* ── hero ── */
.hero{
  max-width:var(--maxw);margin:0 auto;
  padding:72px 24px 40px;
  display:flex;flex-direction:column;align-items:center;text-align:center;
}
.hero-eyebrow{
  display:inline-flex;align-items:center;gap:9px;
  font-family:var(--font-mono);font-size:12.5px;letter-spacing:0.04em;
  color:var(--teal);
  padding:8px 16px;border-radius:999px;
  background:rgba(94,234,212,0.08);border:1px solid rgba(94,234,212,0.2);
  margin-bottom:30px;
}
.pulse-dot{
  width:7px;height:7px;border-radius:50%;background:var(--teal);
  box-shadow:0 0 0 0 rgba(94,234,212,0.6);animation:pulse 2s infinite;
}
@keyframes pulse{0%{box-shadow:0 0 0 0 rgba(94,234,212,0.5);}70%{box-shadow:0 0 0 8px rgba(94,234,212,0);}100%{box-shadow:0 0 0 0 rgba(94,234,212,0);}}

.hero-title{
  font-family:var(--font-display);font-weight:700;
  font-size:clamp(56px,12vw,128px);line-height:0.95;letter-spacing:-0.04em;
  color:var(--ink);
  text-shadow:0 0 60px rgba(56,189,248,0.25);
}
.hero-in{
  background:linear-gradient(135deg,var(--teal),var(--sky));
  -webkit-background-clip:text;background-clip:text;-webkit-text-fill-color:transparent;
}
.hero-sub{
  max-width:620px;margin:28px auto 0;
  font-size:18px;line-height:1.6;color:var(--ink-soft);font-weight:400;
}
.hero-actions{display:flex;gap:14px;margin-top:36px;flex-wrap:wrap;justify-content:center;}

.btn-primary{
  display:inline-flex;align-items:center;gap:9px;
  font-family:var(--font-body);font-weight:600;font-size:15.5px;
  color:#04181A;background:linear-gradient(135deg,var(--teal),var(--sky));
  border:none;border-radius:12px;padding:14px 26px;cursor:pointer;
  transition:transform .2s, box-shadow .2s;
}
.btn-primary:hover{transform:translateY(-2px);box-shadow:0 12px 32px rgba(56,189,248,0.4);}
.btn-primary.lg{font-size:16.5px;padding:16px 32px;}
.btn-ghost{
  display:inline-flex;align-items:center;
  font-weight:600;font-size:15.5px;color:var(--ink);
  text-decoration:none;border-radius:12px;padding:14px 26px;
  background:var(--glass);border:1px solid var(--glass-border-soft);
  backdrop-filter:blur(12px);transition:border-color .2s, background .2s;
}
.btn-ghost:hover{border-color:var(--glass-border);background:var(--glass-strong);}

/* hero demo */
.hero-demo{
  margin-top:56px;width:100%;max-width:560px;text-align:left;
  padding:24px 26px;
}
.demo-head{display:flex;justify-content:space-between;align-items:center;margin-bottom:14px;}
.demo-q-label{
  font-family:var(--font-mono);font-size:11px;letter-spacing:0.08em;
  text-transform:uppercase;color:var(--ink-faint);
}
.demo-provider{
  font-family:var(--font-mono);font-size:11px;color:var(--teal);
  padding:4px 10px;border-radius:999px;background:rgba(94,234,212,0.1);
}
.demo-q{font-family:var(--font-display);font-size:19px;font-weight:600;color:var(--ink);line-height:1.4;}
.demo-divider{height:1px;background:var(--glass-border-soft);margin:18px 0;}
.demo-a{font-size:15px;line-height:1.7;color:var(--ink-soft);}
.cite{
  display:inline-flex;align-items:center;justify-content:center;
  font-family:var(--font-mono);font-size:10px;font-weight:600;
  min-width:16px;height:16px;padding:0 4px;margin:0 2px;
  border-radius:5px;color:var(--teal);
  background:rgba(94,234,212,0.14);border:1px solid rgba(94,234,212,0.3);
  vertical-align:super;
}
.demo-sources{display:flex;gap:8px;margin-top:18px;}
.src-chip{
  font-family:var(--font-mono);font-size:11px;color:var(--ink-soft);
  padding:5px 12px;border-radius:8px;
  background:rgba(255,255,255,0.05);border:1px solid var(--glass-border-soft);
}

/* ── stats ── */
.stats{
  max-width:var(--maxw);margin:48px auto 0;padding:0 24px;
  display:grid;grid-template-columns:repeat(4,1fr);gap:16px;
}
.stat{padding:28px 24px;text-align:center;}
.stat-num{
  font-family:var(--font-display);font-weight:700;font-size:44px;
  color:var(--ink);letter-spacing:-0.02em;line-height:1;
}
.stat-pct{font-size:26px;color:var(--teal);}
.stat-label{margin-top:10px;font-size:13.5px;color:var(--ink-faint);font-weight:500;}

/* ── sections ── */
.section{max-width:var(--maxw);margin:0 auto;padding:96px 24px 0;}
.section-head{text-align:center;max-width:660px;margin:0 auto 48px;}
.eyebrow{
  font-family:var(--font-mono);font-size:12.5px;letter-spacing:0.12em;
  text-transform:uppercase;color:var(--teal);
}
.section-title{
  font-family:var(--font-display);font-weight:700;
  font-size:clamp(30px,4.5vw,44px);letter-spacing:-0.025em;color:var(--ink);
  margin-top:14px;line-height:1.1;
}
.section-lead{margin-top:16px;font-size:17px;line-height:1.6;color:var(--ink-soft);}

/* steps */
.steps{display:grid;grid-template-columns:repeat(4,1fr);gap:18px;}
.step{padding:28px 24px;opacity:0;transform:translateY(24px);}
.steps.in .step{animation:rise .7s cubic-bezier(.2,.7,.2,1) forwards;animation-delay:var(--d);}
.step-index{
  font-family:var(--font-mono);font-size:13px;color:var(--teal);
  letter-spacing:0.05em;margin-bottom:16px;
}
.step-title{font-family:var(--font-display);font-weight:600;font-size:20px;color:var(--ink);margin-bottom:10px;}
.step-body{font-size:14.5px;line-height:1.6;color:var(--ink-soft);}

/* cards */
.cards{display:grid;grid-template-columns:repeat(3,1fr);gap:18px;}
.card{padding:30px 28px;opacity:0;transform:translateY(24px);}
.cards.in .card{animation:rise .7s cubic-bezier(.2,.7,.2,1) forwards;animation-delay:var(--d);}
.card-tag{
  display:inline-block;font-family:var(--font-mono);font-size:11px;
  letter-spacing:0.08em;text-transform:uppercase;color:var(--teal);
  padding:5px 12px;border-radius:999px;
  background:rgba(94,234,212,0.1);border:1px solid rgba(94,234,212,0.22);
  margin-bottom:18px;
}
.card-title{font-family:var(--font-display);font-weight:700;font-size:28px;color:var(--ink);margin-bottom:12px;}
.card-body{font-size:14.5px;line-height:1.6;color:var(--ink-soft);min-height:66px;}
.card-count{
  margin-top:18px;font-family:var(--font-mono);font-size:13px;color:var(--ink-faint);
  padding-top:16px;border-top:1px solid var(--glass-border-soft);
}

/* why */
.why-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:18px;}
.why-item{padding:28px 26px;opacity:0;transform:translateY(24px);}
.why-grid.in .why-item{animation:rise .7s cubic-bezier(.2,.7,.2,1) forwards;animation-delay:var(--d);}
.why-icon{
  font-size:24px;width:52px;height:52px;border-radius:14px;
  display:flex;align-items:center;justify-content:center;margin-bottom:18px;
  background:linear-gradient(135deg,rgba(94,234,212,0.16),rgba(56,189,248,0.12));
  border:1px solid var(--glass-border);
}
.why-title{font-family:var(--font-display);font-weight:600;font-size:18px;color:var(--ink);margin-bottom:10px;}
.why-body{font-size:14px;line-height:1.6;color:var(--ink-soft);}

@keyframes rise{to{opacity:1;transform:translateY(0);}}

/* cta */
.cta-band{max-width:var(--maxw);margin:110px auto 0;padding:0 24px;}
.cta-card{
  padding:64px 40px;text-align:center;
  background:linear-gradient(135deg,rgba(94,234,212,0.10),rgba(56,189,248,0.06));
  border:1px solid var(--glass-border);
}
.cta-title{font-family:var(--font-display);font-weight:700;font-size:clamp(28px,4vw,40px);letter-spacing:-0.025em;color:var(--ink);}
.cta-sub{margin-top:14px;font-size:17px;color:var(--ink-soft);}
.cta-card .btn-primary{margin-top:30px;}

/* footer */
.footer{margin-top:110px;padding:0 24px;}
.footer-inner{
  max-width:var(--maxw);margin:0 auto;padding:56px 40px 40px;
  display:grid;grid-template-columns:2fr 1fr 1fr 1.4fr;gap:40px;
  background:rgba(8,22,26,0.5);
  backdrop-filter:blur(18px);-webkit-backdrop-filter:blur(18px);
  border:1px solid var(--glass-border-soft);
  border-radius:24px 24px 0 0;
}
.footer-brand-col .brand{margin-bottom:16px;}
.footer-tagline{font-size:14px;line-height:1.6;color:var(--ink-soft);max-width:300px;}
.footer-col{display:flex;flex-direction:column;}
.footer-h{
  font-family:var(--font-display);font-size:13px;font-weight:600;
  letter-spacing:0.06em;text-transform:uppercase;color:var(--ink-faint);
  margin-bottom:16px;
}
.footer-col a{
  color:var(--ink-soft);text-decoration:none;font-size:14px;font-weight:500;
  margin-bottom:11px;transition:color .2s;
}
.footer-col a:hover{color:var(--teal);}
.footer-static{font-size:14px;color:var(--ink-faint);margin-bottom:11px;}

.footer-limits{
  max-width:var(--maxw);margin:0 auto;padding:22px 40px;
  background:rgba(8,22,26,0.5);
  backdrop-filter:blur(18px);-webkit-backdrop-filter:blur(18px);
  border-left:1px solid var(--glass-border-soft);
  border-right:1px solid var(--glass-border-soft);
}
.footer-limits p{font-size:12.5px;line-height:1.7;color:var(--ink-faint);}
.footer-limits strong{color:var(--ink-soft);font-weight:600;}

.footer-base{
  max-width:var(--maxw);margin:0 auto;padding:20px 40px 40px;
  display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:10px;
  background:rgba(8,22,26,0.5);
  backdrop-filter:blur(18px);-webkit-backdrop-filter:blur(18px);
  border-left:1px solid var(--glass-border-soft);
  border-right:1px solid var(--glass-border-soft);
  border-bottom:1px solid var(--glass-border-soft);
  border-radius:0 0 24px 24px;
  font-size:12.5px;color:var(--ink-faint);
}
.footer-base-note{font-family:var(--font-mono);font-size:11.5px;color:var(--ink-faint);}

/* responsive */
@media(max-width:900px){
  .stats{grid-template-columns:repeat(2,1fr);}
  .steps{grid-template-columns:repeat(2,1fr);}
  .cards{grid-template-columns:1fr;}
  .why-grid{grid-template-columns:1fr;}
  .footer-inner{grid-template-columns:1fr 1fr;gap:32px;}
  .nav-links a{display:none;}
}
@media(max-width:560px){
  .stats{grid-template-columns:1fr 1fr;}
  .steps{grid-template-columns:1fr;}
  .footer-inner{grid-template-columns:1fr;}
  .footer-base{flex-direction:column;align-items:flex-start;}
  .hero{padding-top:48px;}
}

@media(prefers-reduced-motion:reduce){
  *{animation:none!important;transition:none!important;}
  .step,.card,.why-item{opacity:1!important;transform:none!important;}
}
`;
