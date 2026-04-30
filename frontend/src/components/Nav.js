import { Link } from 'react-router-dom';
import './Nav.css';

export default function Nav() {
  return (
    <nav className="nav">
      <Link to="/" className="nav-brand">
        <span className="brand-mark">H</span>
        HealthRAG<span style={{color:'var(--cyan)'}}>-IN</span>
      </Link>
      <div className="nav-links">
        <a href="#how">How it works</a>
        <a href="#sources">Sources</a>
        <a href="#why">Why</a>
      </div>
      <Link to="/chat" className="nav-cta">Begin Consultation →</Link>
    </nav>
  );
}