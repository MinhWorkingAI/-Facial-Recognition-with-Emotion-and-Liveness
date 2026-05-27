import React from 'react';
import { EmotionGlyph } from './Icons.jsx';

export default function DetectionPanel({ analysis }) {
  const faces = analysis?.faces || [];
  const face = faces[0];
  const recognition = face?.recognition;
  const emotion = face?.emotion;
  const antiSpoofing = face?.anti_spoofing;
  const noFace = !face;
  const multiFace = faces.length > 1;

  return (
    <section className="section">
      <header className="section__head">
        <div className="section__head-title">
          <span className="section__num">§ 02</span>
          <span className="section__title">People on Screen</span>
        </div>
        <span className="section__aside">
          {noFace
            ? 'Live readings'
            : multiFace
              ? <><span style={{ color: 'var(--accent)' }}>{faces.length} faces</span> · showing №1</>
              : '1 face · live'}
        </span>
      </header>

      {multiFace && (
        <div className="multi-face-note">
          Stats for the remaining {faces.length - 1} {faces.length - 1 === 1 ? 'face' : 'faces'} are
          drawn next to {faces.length - 1 === 1 ? 'their box' : 'their boxes'} in the webcam view.
        </div>
      )}

      {/* IDENTITY */}
      <div className="readout">
        <div className="readout__label">Identity</div>
        {noFace ? (
          <div className="readout__value readout__value--mute">No one in frame</div>
        ) : recognition?.matched && recognition.label !== 'unknown' ? (
          <>
            <div className="readout__value readout__value--accent">{recognition.label}</div>
            <ConfidenceBar value={recognition.confidence} variant="accent" />
          </>
        ) : (
          <>
            <div className="readout__value readout__value--mute">Unregistered</div>
            <div className="readout__sub">
              <span>No match in registry</span>
              <span className="t-num">
                {recognition ? `${(recognition.confidence * 100).toFixed(1)}%` : '—'}
              </span>
            </div>
          </>
        )}
      </div>

      {/* EMOTION */}
      <div className="readout">
        <div className="readout__label">Affect</div>
        {noFace ? (
          <div className="readout__value readout__value--mute">—</div>
        ) : (
          <>
            <div className="readout__row">
              <EmotionGlyph emotion={emotion?.label} />
              <div style={{ flex: 1 }}>
                <div className="readout__value">{capitalize(emotion?.label) || '—'}</div>
              </div>
            </div>
            <ConfidenceBar value={emotion?.confidence ?? 0} />
          </>
        )}
      </div>

      {/* LIVENESS */}
      <div className="readout">
        <div className="readout__label">Liveness</div>
        {noFace ? (
          <div className="readout__value readout__value--mute">—</div>
        ) : antiSpoofing?.label === 'spoof' ? (
          <>
            <div className="readout__row">
              <span className="readout__indicator stop" />
              <span className="readout__value readout__value--stop">Spoof Detected</span>
            </div>
            <ConfidenceBar value={antiSpoofing.confidence} variant="accent" />
          </>
        ) : antiSpoofing?.label === 'real' ? (
          <>
            <div className="readout__row">
              <span className="readout__indicator go" />
              <span className="readout__value readout__value--go">Live</span>
            </div>
            <ConfidenceBar value={antiSpoofing.confidence} variant="go" />
          </>
        ) : (
          <div className="readout__value readout__value--mute">Awaiting subject</div>
        )}
      </div>
    </section>
  );
}

function capitalize(s) {
  if (!s) return '';
  return s.charAt(0).toUpperCase() + s.slice(1).toLowerCase();
}

function ConfidenceBar({ value = 0, variant }) {
  const pct = Math.max(0, Math.min(1, value));
  const fillClass = variant ? `readout__sub-bar-fill--${variant}` : '';
  return (
    <div className="readout__sub">
      <span>Confidence</span>
      <span className="readout__sub-bar">
        <span className={`readout__sub-bar-fill ${fillClass}`} style={{ width: `${pct * 100}%` }} />
      </span>
      <span className="t-num">{(pct * 100).toFixed(1)}%</span>
    </div>
  );
}