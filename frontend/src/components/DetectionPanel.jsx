import React from 'react';
import { EmotionGlyph } from './Icons.jsx';

export default function DetectionPanel({ analysis }) {
  const faces = analysis?.faces || [];
  const noFace = faces.length === 0;
  const singleFace = faces.length === 1;
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
              ? <><span style={{ color: 'var(--accent)' }}>{faces.length} faces</span> · roster</>
              : '1 face · live'}
        </span>
      </header>

      {noFace && <EmptyState />}
      {singleFace && <SingleFaceView face={faces[0]} />}
      {multiFace && <MultiFaceRoster faces={faces} />}
    </section>
  );
}

/* Empty state */
function EmptyState() {
  return (
    <>
      <div className="readout">
        <div className="readout__label">Identity</div>
        <div className="readout__value readout__value--mute">No one in frame</div>
      </div>
      <div className="readout">
        <div className="readout__label">Affect</div>
        <div className="readout__value readout__value--mute">—</div>
      </div>
      <div className="readout">
        <div className="readout__label">Liveness</div>
        <div className="readout__value readout__value--mute">—</div>
      </div>
    </>
  );
}

/* Single-face detailed view */
function SingleFaceView({ face }) {
  const recognition = face?.recognition;
  const emotion = face?.emotion;
  const antiSpoofing = face?.anti_spoofing;

  return (
    <>
      <div className="readout">
        <div className="readout__label">Identity</div>
        {recognition?.matched && recognition.label !== 'unknown' ? (
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

      <div className="readout">
        <div className="readout__label">Affect</div>
        <div className="readout__row">
          <EmotionGlyph emotion={emotion?.label} />
          <div style={{ flex: 1 }}>
            <div className="readout__value">{capitalize(emotion?.label) || '—'}</div>
          </div>
        </div>
        <ConfidenceBar value={emotion?.confidence ?? 0} />
      </div>

      <div className="readout">
        <div className="readout__label">Liveness</div>
        {antiSpoofing?.label === 'spoof' ? (
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
    </>
  );
}

/* Multi-face roster */
function MultiFaceRoster({ faces }) {
  return (
    <ul className="face-roster">
      {faces.map((face, i) => {
        const rec = face.recognition;
        const emo = face.emotion;
        const live = face.anti_spoofing;
        const matched = rec?.matched && rec.label !== 'unknown';

        return (
          <li key={i} className="face-roster__item">
            <div className="face-roster__head">
              <span className="face-roster__num">№{String(i + 1).padStart(2, '0')}</span>
              <span className={`face-roster__name ${matched ? 'matched' : 'unmatched'}`}>
                {matched ? rec.label : 'Unidentified'}
              </span>
              {rec && (
                <span className="face-roster__conf">
                  {(rec.confidence * 100).toFixed(0)}%
                </span>
              )}
            </div>

            <div className="face-roster__stats">
              {emo?.label && (
                <span className="face-roster__stat">
                  {capitalize(emo.label)}
                  {emo.confidence != null && (
                    <span className="face-roster__pct"> {(emo.confidence * 100).toFixed(0)}%</span>
                  )}
                </span>
              )}
              {live?.label === 'real' && (
                <span className="face-roster__stat go">
                  <span className="face-roster__dot go" />
                  LIVE{live.confidence != null ? ` ${(live.confidence * 100).toFixed(0)}%` : ''}
                </span>
              )}
              {live?.label === 'spoof' && (
                <span className="face-roster__stat stop">
                  <span className="face-roster__dot stop" />
                  SPOOF{live.confidence != null ? ` ${(live.confidence * 100).toFixed(0)}%` : ''}
                </span>
              )}
            </div>
          </li>
        );
      })}
    </ul>
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