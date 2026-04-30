import { Link } from 'react-router-dom';

export default function Chat() {
  return (
    <div style={{ padding: '120px 48px', textAlign: 'center' }}>
      <h1 style={{ fontFamily: 'var(--display)', fontSize: '64px', color: 'var(--cream)' }}>
        Chat coming next…
      </h1>
      <p style={{ marginTop: 24, color: 'var(--cream-2)' }}>
        Step R4 will build this.
      </p>
      <Link to="/" style={{ color: 'var(--cyan)', marginTop: 32, display: 'inline-block' }}>
        ← Back to landing
      </Link>
    </div>
  );
}