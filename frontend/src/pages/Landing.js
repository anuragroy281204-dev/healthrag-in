import { Link } from 'react-router-dom';
import { motion } from 'framer-motion';
import { useEffect, useState } from 'react';
import Nav from '../components/Nav';
import './Landing.css';

const fadeUp = {
  hidden: { opacity: 0, y: 24 },
  visible: { opacity: 1, y: 0, transition: { duration: 0.8, ease: 'easeOut' } }
};

const stagger = {
  visible: { transition: { staggerChildren: 0.12 } }
};

function Counter({ target }) {
  const [val, setVal] = useState(0);
  useEffect(() => {
    const duration = 1800;
    const start = performance.now();
    let raf;
    const step = (now) => {
      const t = Math.min((now - start) / duration, 1);
      const eased = 1 - Math.pow(1 - t, 3);
      setVal(Math.floor(eased * target));
      if (t < 1) raf = requestAnimationFrame(step);
    };
    raf = requestAnimationFrame(step);
    return () => cancelAnimationFrame(raf);
  }, [target]);
  return <>{val.toLocaleString()}</>;
}

const PIPELINE_STEPS = [
  {
    num: '01 — Retrieve',
    title: 'Hybrid Search',
    body: 'Your question is searched across 1,247 medical chunks using two retrieval methods in parallel. FAISS finds chunks with similar meaning via 384-dimensional embeddings. BM25 finds chunks with the exact terminology you used. Both result lists are fused via Reciprocal Rank Fusion to produce a single ranked top-5.',
    tags: ['FAISS', 'BM25', 'RRF']
  },
  {
    num: '02 — Ground',
    title: 'Strict Prompt',
    body: 'The retrieved chunks are packaged into a versioned system prompt with three non-negotiable rules: use only these sources, cite every factual claim, refuse explicitly if the sources cannot support the answer. The prompt also blocks personalized medical advice and detects emergencies.',
    tags: ['Prompt v1.0.0', 'Refusal-aware', 'Safety guardrails']
  },
  {
    num: '03 — Generate',
    title: 'Resilient Inference',
    body: 'Llama 3.3 70B (via Groq) writes the answer at temperature 0.1 for deterministic, source-faithful output. If Groq is rate-limited or unavailable, the system automatically falls back to Google Gemini 2.0 Flash. Same prompt, same output contract — only the underlying model changes.',
    tags: ['Llama 3.3 70B', 'Gemini 2.0 Flash', 'Auto-failover']
  },
  {
    num: '04 — Cite',
    title: 'Verifiable Output',
    body: 'Every factual claim ends with a [N] marker. Each marker maps to a real chunk with its source name, document title, and clickable URL. A citation validator checks that no marker points to a chunk that was not actually retrieved — catching hallucinated citations before they reach the user.',
    tags: ['100% verifiable', 'Clickable sources', 'No hallucinations']
  }
];

const SOURCES = [
  {
    logo: 'WHO',
    name: 'World Health Organization',
    stat: '04 / docs',
    body: 'Authoritative consumer-facing fact sheets covering diabetes, hypertension, obesity, and cardiovascular disease. Used as the baseline answer source for general questions.',
    tags: ['Diabetes', 'Hypertension', 'Obesity', 'CVD'],
    url: 'https://www.who.int/news-room/fact-sheets',
    feature: false
  },
  {
    logo: 'PubMed',
    name: 'National Library of Medicine',
    stat: '480 / abstracts',
    body: 'Peer-reviewed diabetes research from 2020 onward, fetched via NCBI E-utilities. Includes Indian-population studies, clinical trials, and emerging evidence reviews.',
    tags: ['Type 1 / 2', 'Gestational', 'Indian cohorts', '2020–2026'],
    url: 'https://pubmed.ncbi.nlm.nih.gov',
    feature: true
  },
  {
    logo: 'ICMR',
    name: 'Indian Council of Medical Research',
    stat: '06 / guidelines',
    body: 'India-specific clinical guidelines and Standard Treatment Workflows. Covers Type 1, Type 2, diabetic foot, ketoacidosis, and the 2024 NIN Dietary Guidelines.',
    tags: ['STW', 'Diabetic foot', 'DKA', 'NIN 2024'],
    url: 'https://www.icmr.gov.in',
    feature: false
  }
];

const WHY_CARDS = [
  { icon: '◎', title: 'No hallucinations', body: 'The system literally cannot invent facts — it can only synthesize what was retrieved. Every claim must trace back to a chunk in the corpus or be refused.' },
  { icon: '⌖', title: 'Refusal-first', body: 'Out-of-scope and personal-advice questions are explicitly refused, not guessed at. Two layers: code-level relevance gate plus prompt-level rule.' },
  { icon: '▣', title: 'India-aware', body: 'ICMR guidelines and Indian PubMed studies are surfaced when relevant. HbA1c targets, dosing, and dietary recommendations reflect Indian context.' },
  { icon: '◇', title: 'Provider-resilient', body: 'Multi-provider LLM routing with automatic failover. Groq Llama 3.3 70B primary, Gemini 2.0 Flash secondary. Stays online when one provider rate-limits.' },
  { icon: '⊞', title: 'Measured, not claimed', body: 'A five-metric LLM-as-judge evaluation harness scores every answer for faithfulness, relevance, citation accuracy, and refusal correctness.' },
  { icon: '⌬', title: 'Open & free', body: 'Built entirely on free-tier APIs and open-source libraries. Source available on GitHub. Zero rupee operational cost.' }
];

export default function Landing() {
  return (
    <>
      <Nav />
      <div className="bg-grid"></div>
      <div className="bg-glow"></div>

      {/* HERO */}
      <section className="hero" id="top">
        <motion.div
          className="medical-cross"
          initial={{ scale: 0, rotate: -180, opacity: 0 }}
          animate={{ scale: 1, rotate: 0, opacity: 1 }}
          transition={{ duration: 1.4, ease: [0.16, 1, 0.3, 1] }}
          whileHover={{ scale: 1.1, rotate: 90, transition: { duration: 0.6 } }}
        >
          <div className="cross-ring"></div>
          <div className="cross-shape">
            <div className="cross-h"></div>
            <div className="cross-v"></div>
          </div>
          <div className="cross-glow"></div>
        </motion.div>

        <div className="hero-content">
          <motion.div
            className="hero-eyebrow"
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 1.6 }}
          >
            <span className="eyebrow-dot"></span>
            Llama 3.3 · Gemini 2.0 · Live
          </motion.div>

          <motion.h1
            className="hero-title"
            initial={{ opacity: 0, y: 24 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 1, delay: 1.8 }}
          >
            Medicine,<br /><em>cited.</em>
          </motion.h1>

          <motion.p
            className="hero-sub"
            initial={{ opacity: 0, y: 24 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 1, delay: 2.0 }}
          >
            A grounded medical Q&amp;A system for diabetes — answers come <strong>only</strong> from
            real clinical evidence, every claim is cited, and personal medical questions are
            redirected to professionals.
          </motion.p>

          <motion.div
            className="hero-actions"
            initial={{ opacity: 0, y: 24 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 1, delay: 2.2 }}
          >
            <Link to="/chat" className="btn btn-primary">
              Begin Consultation <span className="btn-arrow">→</span>
            </Link>
            <a
              href="https://github.com/anuragroy281204-dev/healthrag-in"
              target="_blank"
              rel="noreferrer"
              className="btn btn-secondary"
            >
              View on GitHub
            </a>
          </motion.div>
        </div>

        <div className="citation-rail citation-rail-1">
          <motion.div className="cite-chip" initial={{ opacity: 0, x: -20 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: 2.6, duration: 0.6 }}>
            <span className="cite-chip-num">[1]</span> WHO Fact Sheet
          </motion.div>
          <motion.div className="cite-chip" initial={{ opacity: 0, x: -20 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: 2.8, duration: 0.6 }}>
            <span className="cite-chip-num">[2]</span> ICMR Guidelines
          </motion.div>
          <motion.div className="cite-chip" initial={{ opacity: 0, x: -20 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: 3.0, duration: 0.6 }}>
            <span className="cite-chip-num">[3]</span> PubMed 2024
          </motion.div>
        </div>

        <div className="citation-rail citation-rail-2">
          <motion.div className="cite-chip" initial={{ opacity: 0, x: 20 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: 2.6, duration: 0.6 }}>
            <span className="cite-chip-num">[4]</span> NIN Dietary 2024
          </motion.div>
          <motion.div className="cite-chip" initial={{ opacity: 0, x: 20 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: 2.8, duration: 0.6 }}>
            <span className="cite-chip-num">[5]</span> ICMR STW
          </motion.div>
          <motion.div className="cite-chip" initial={{ opacity: 0, x: 20 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: 3.0, duration: 0.6 }}>
            <span className="cite-chip-num">[6]</span> Lancet Diabetes
          </motion.div>
        </div>

        <motion.div
          className="hero-stats"
          initial={{ opacity: 0, y: 24 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 1, delay: 2.4 }}
        >
          <div className="stat-block">
            <div className="stat-num"><Counter target={490} /></div>
            <div className="stat-label">Documents</div>
          </div>
          <div className="stat-block">
            <div className="stat-num"><Counter target={1247} /></div>
            <div className="stat-label">Chunks indexed</div>
          </div>
          <div className="stat-block">
            <div className="stat-num">3</div>
            <div className="stat-label">Sources</div>
          </div>
          <div className="stat-block">
            <div className="stat-num">5</div>
            <div className="stat-label">Eval metrics</div>
          </div>
        </motion.div>
      </section>

      {/* HOW IT WORKS */}
      <section id="how">
        <div className="container">
          <motion.div
            initial={{ opacity: 0, y: 16 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true, amount: 0.3 }}
            transition={{ duration: 0.6 }}
          >
            <div className="section-eyebrow">The Pipeline</div>
            <h2 className="section-title">From your <em>question</em><br />to a cited answer.</h2>
          </motion.div>

          <motion.div
            className="pipeline"
            variants={stagger}
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true, amount: 0.1 }}
          >
            {PIPELINE_STEPS.map((step, i) => (
              <motion.div key={i} className="pipe-step" variants={fadeUp}>
                <div className="pipe-num">{step.num}</div>
                <h3 className="pipe-h">{step.title}</h3>
                <p className="pipe-p">{step.body}</p>
                <div className="pipe-tag-row">
                  {step.tags.map((t) => (
                    <div key={t} className="pipe-tag">{t}</div>
                  ))}
                </div>
              </motion.div>
            ))}
          </motion.div>
        </div>
      </section>

      {/* SOURCES */}
      <section id="sources">
        <div className="container">
          <motion.div
            initial={{ opacity: 0, y: 16 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true, amount: 0.3 }}
            transition={{ duration: 0.6 }}
          >
            <div className="section-eyebrow">The Corpus</div>
            <h2 className="section-title">Three trusted<br /><em>source families.</em></h2>
          </motion.div>

          <motion.div
            className="sources-grid"
            variants={stagger}
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true, amount: 0.1 }}
          >
            {SOURCES.map((src, i) => (
              <motion.div
                key={i}
                className={`source-card ${src.feature ? 'source-card-feature' : ''}`}
                variants={fadeUp}
              >
                <div className="source-logo">
                  <div className="logo-placeholder">{src.logo}</div>
                </div>
                <div className="source-meta">
                  <div className="source-name">{src.name}</div>
                  <div className="source-stat">{src.stat}</div>
                </div>
                <p className="source-p">{src.body}</p>
                <div className="source-tags">
                  {src.tags.map((t) => (
                    <span key={t} className="src-tag">{t}</span>
                  ))}
                </div>
                <a href={src.url} target="_blank" rel="noreferrer" className="source-link">
                  {src.url.replace('https://', '')} →
                </a>
              </motion.div>
            ))}
          </motion.div>
        </div>
      </section>

      {/* WHY */}
      <section id="why">
        <div className="container">
          <motion.div
            initial={{ opacity: 0, y: 16 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true, amount: 0.3 }}
            transition={{ duration: 0.6 }}
          >
            <div className="section-eyebrow">Differentiation</div>
            <h2 className="section-title">Built different,<br /><em>on purpose.</em></h2>
          </motion.div>

          <motion.div
            className="why-grid"
            variants={stagger}
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true, amount: 0.1 }}
          >
            {WHY_CARDS.map((why, i) => (
              <motion.div key={i} className="why-card" variants={fadeUp}>
                <div className="why-icon">{why.icon}</div>
                <h3 className="why-h">{why.title}</h3>
                <p className="why-p">{why.body}</p>
              </motion.div>
            ))}
          </motion.div>
        </div>
      </section>

      {/* CTA */}
      <section id="cta" className="cta-section">
        <div className="container cta-container">
          <motion.h2
            className="cta-title"
            initial={{ opacity: 0, y: 24 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true, amount: 0.4 }}
            transition={{ duration: 0.8 }}
          >
            Ask your first<br /><em>question.</em>
          </motion.h2>
          <p className="cta-p">
            Real medical sources. Real citations. Real refusals when we don't know.
          </p>
          <Link to="/chat" className="btn btn-primary btn-large">
            Begin Consultation <span className="btn-arrow">→</span>
          </Link>
          <p className="cta-disclaimer">
            Educational tool. Not medical advice. Always consult a qualified physician.
          </p>
        </div>
      </section>

      {/* FOOTER */}
      <footer className="footer">
        <div className="container footer-container">
          <div>
            <div className="nav-brand">
              <span className="brand-mark">H</span>
              HealthRAG<span style={{ color: 'var(--cyan)' }}>-IN</span>
            </div>
            <p className="footer-p">
              Built by Anurag Roy — B.Tech CSE (Data Science), Amity University Noida.
            </p>
          </div>
          <div className="footer-links">
            <a href="https://github.com/anuragroy281204-dev/healthrag-in" target="_blank" rel="noreferrer">GitHub</a>
            <a href="mailto:royanurag281204@gmail.com">Contact</a>
          </div>
        </div>
        <div className="footer-bottom">
          © 2026 HealthRAG-IN · Educational use only · Not a substitute for medical advice
        </div>
      </footer>
    </>
  );
}