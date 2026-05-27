import React, { useEffect, useRef, useState } from 'react';

/**
 * Registration modal. Asks for an ID (required) and an optional
 * display name. If no display name is given, the ID is used as the
 * display name. Both are forwarded to the backend's /register endpoint.
 */
export default function RegisterModal({ open, onClose, onConfirm }) {
  const [personId, setPersonId] = useState('');
  const [personName, setPersonName] = useState('');
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);
  const inputRef = useRef(null);

  useEffect(() => {
    if (open) {
      setPersonId('');
      setPersonName('');
      setError('');
      setBusy(false);
      setTimeout(() => inputRef.current?.focus(), 100);
    }
  }, [open]);

  if (!open) return null;

  const submit = async () => {
    const id = personId.trim();
    const name = personName.trim();
    if (!id) {
      setError('An ID is required.');
      return;
    }
    setBusy(true);
    setError('');
    try {
      await onConfirm(id, name || id);
    } catch (e) {
      setError(e.message || 'Registration failed.');
      setBusy(false);
    }
  };

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal" onClick={(e) => e.stopPropagation()}>
        <div className="modal__num">§ Enrol face</div>
        <h2 className="modal__title">Enrol a new <em>face</em></h2>
        <p className="modal__body">
          Hold steady in front of the camera please.
        </p>

        {error && <div className="modal__error">// {error}</div>}

        <label className="modal__field-label">ID <span>required</span></label>
        <input
          ref={inputRef}
          className="modal__input"
          type="text"
          value={personId}
          placeholder="e.g. emp_021"
          onChange={(e) => setPersonId(e.target.value)}
          onKeyDown={(e) => { if (e.key === 'Enter') submit(); }}
          disabled={busy}
          autoComplete="off"
        />

        <label className="modal__field-label">Display name <span>optional</span></label>
        <input
          className="modal__input"
          type="text"
          value={personName}
          placeholder="leave blank to match the ID"
          onChange={(e) => setPersonName(e.target.value)}
          onKeyDown={(e) => { if (e.key === 'Enter') submit(); }}
          disabled={busy}
          autoComplete="off"
        />

        <div className="modal__actions">
          <button className="btn btn--ghost" onClick={onClose} disabled={busy}>
            Cancel
          </button>
          <button className="btn btn--primary" onClick={submit} disabled={busy}>
            {busy ? 'Recording…' : 'Register'}
          </button>
        </div>
      </div>
    </div>
  );
}
