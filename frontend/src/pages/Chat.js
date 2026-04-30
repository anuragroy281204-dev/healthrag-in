import { useState, useEffect, useRef } from 'react';
import { Link } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import './Chat.css';

const API_URL = 'http://localhost:8000';

const SAMPLE_QUESTIONS = [
  'What is HbA1c and why does it matter?',
  'What diet do ICMR guidelines recommend for diabetics in India?',
  'What is the difference between Type 1 and Type 2 diabetes?',
  'How is diabetic ketoacidosis treated?',
];

export default function Chat() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [stats, setStats] = useState(null);
  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);

  useEffect(() => {
    fetch(`${API_URL}/stats`)
      .then(r => r.json())
      .then(setStats)
      .catch(() => setStats(null));
  }, []);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, loading]);

  const sendQuestion = async (question) => {
    if (!question.trim() || loading) return;

    const userMsg = { role: 'user', content: question, id: Date.now() };
    setMessages(prev => [...prev, userMsg]);
    setInput('');
    setLoading(true);

    try {
      const res = await fetch(`${API_URL}/ask`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question }),
      });

      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: 'Request failed' }));
        throw new Error(err.detail || `HTTP ${res.status}`);
      }

      const data = await res.json();
      const assistantMsg = {
        role: 'assistant',
        id: Date.now() + 1,
        ...data,
      };
      setMessages(prev => [...prev, assistantMsg]);
    } catch (err) {
      setMessages(prev => [...prev, {
        role: 'error',
        id: Date.now() + 1,
        content: err.message || 'Something went wrong. Make sure the backend is running on port 8000.',
      }]);
    } finally {
      setLoading(false);
      inputRef.current?.focus();
    }
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    sendQuestion(input);
  };

  const renderAnswerWithChips = (text) => {
    const parts = text.split(/(\[[\d,\s]+\])/g);
    return parts.map((part, i) => {
      if (/^\[[\d,\s]+\]$/.test(part)) {
        return <span key={i} className="cite-pill">{part}</span>;
      }
      return <span key={i}>{part}</span>;
    });
  };

  return (
    <div className="chat-page">
      <div className="bg-grid"></div>
      <div className="bg-glow"></div>

      <header className="chat-header">
        <Link to="/" className="chat-brand">
          <span className="brand-mark">H</span>
          HealthRAG<span style={{color:'var(--cyan)'}}>-IN</span>
        </Link>
        <div className="chat-header-meta">
          {stats && (
            <span className="header-stat">
              <span className="header-stat-num">{stats.chunks?.toLocaleString()}</span> chunks indexed
            </span>
          )}
          <Link to="/" className="header-back">← Landing</Link>
        </div>
      </header>

      <main className="chat-main">

        {messages.length === 0 && (
          <motion.div
            className="chat-welcome"
            initial={{ opacity: 0, y: 24 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
          >
            <div className="welcome-eyebrow">
              <span className="eyebrow-dot"></span>
              Connected · Ready
            </div>
            <h1 className="welcome-title">
              Ask a <em>medical</em> question.
            </h1>
            <p className="welcome-sub">
              Answers come only from WHO fact sheets, PubMed research, and ICMR guidelines.
              Every claim is cited.
            </p>

            <div className="sample-grid">
              {SAMPLE_QUESTIONS.map((q, i) => (
                <motion.button
                  key={q}
                  className="sample-card"
                  onClick={() => sendQuestion(q)}
                  initial={{ opacity: 0, y: 16 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.2 + i * 0.08 }}
                  whileHover={{ y: -4 }}
                >
                  <span className="sample-arrow">→</span>
                  <span className="sample-text">{q}</span>
                </motion.button>
              ))}
            </div>
          </motion.div>
        )}

        <div className="messages">
          <AnimatePresence>
            {messages.map((msg) => (
              <motion.div
                key={msg.id}
                className={`message message-${msg.role}`}
                initial={{ opacity: 0, y: 16 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.4 }}
              >
                {msg.role === 'user' && (
                  <>
                    <div className="msg-label">You asked</div>
                    <div className="msg-bubble msg-user">{msg.content}</div>
                  </>
                )}

                {msg.role === 'assistant' && (
                  <>
                    {msg.is_emergency && (
                      <div className="emergency-banner">
                        ⚠ Possible medical emergency detected. Please call your local
                        emergency number or go to the nearest hospital immediately.
                      </div>
                    )}

                    <div className="msg-label">HealthRAG-IN</div>
                    <div className="msg-bubble msg-assistant">
                      <div className="answer-text">
                        {msg.answer.split('\n').map((line, i) => (
                          <p key={i}>{renderAnswerWithChips(line)}</p>
                        ))}
                      </div>

                      <div className="answer-meta">
                        <span className={`provider-badge provider-${msg.provider}`}>
                          {msg.provider?.toUpperCase() || 'UNKNOWN'}
                        </span>
                        <span className="latency-badge">
                          {msg.total_time_sec?.toFixed(1)}s
                        </span>
                        {msg.is_refusal && (
                          <span className="refusal-badge">REFUSED</span>
                        )}
                      </div>

                      {msg.retrieved_chunks?.length > 0 && (
                        <details className="sources-section">
                          <summary className="sources-summary">
                            <span>📚 Sources ({msg.retrieved_chunks.length})</span>
                            <span className="sources-arrow">▾</span>
                          </summary>
                          <div className="sources-list">
                            {msg.retrieved_chunks.map((chunk, i) => (
                              <div key={i} className="source-item">
                                <div className="source-item-head">
                                  <span className="source-num-pill">{i + 1}</span>
                                  <span className="source-name-tag">{chunk.source}</span>
                                  <span className="source-score">
                                    score {chunk.score?.toFixed(3)}
                                  </span>
                                </div>
                                <div className="source-item-title">{chunk.title}</div>
                                <div className="source-item-text">
                                  {chunk.text?.slice(0, 240)}…
                                </div>
                                {chunk.url && (
                                  <a
                                    href={chunk.url}
                                    target="_blank"
                                    rel="noreferrer"
                                    className="source-item-link"
                                  >
                                    {chunk.url} →
                                  </a>
                                )}
                              </div>
                            ))}
                          </div>
                        </details>
                      )}
                    </div>
                  </>
                )}

                {msg.role === 'error' && (
                  <div className="msg-bubble msg-error">
                    <strong>Error:</strong> {msg.content}
                  </div>
                )}
              </motion.div>
            ))}
          </AnimatePresence>

          {loading && (
            <motion.div
              className="thinking"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
            >
              <span className="thinking-dot"></span>
              <span className="thinking-dot"></span>
              <span className="thinking-dot"></span>
              <span className="thinking-text">
                Searching medical sources and generating answer…
              </span>
            </motion.div>
          )}

          <div ref={messagesEndRef} />
        </div>
      </main>

      <form className="chat-input-wrap" onSubmit={handleSubmit}>
        <div className="chat-input-inner">
          <input
            ref={inputRef}
            type="text"
            className="chat-input"
            placeholder="Ask a medical question about diabetes…"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            disabled={loading}
            autoFocus
          />
          <button
            type="submit"
            className="chat-send"
            disabled={loading || !input.trim()}
          >
            Send <span className="btn-arrow">→</span>
          </button>
        </div>
        <div className="chat-disclaimer">
          Educational tool. Not medical advice. Consult a qualified physician.
        </div>
      </form>
    </div>
  );
}