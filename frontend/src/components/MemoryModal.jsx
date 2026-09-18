import React, { useState } from 'react';

export default function MemoryModal({
  isOpen,
  onClose,
  memories,
  onCreateMemory,
  onDeleteMemory,
}) {
  const [newFact, setNewFact] = useState('');
  const [saving, setSaving] = useState(false);

  if (!isOpen) return null;

  const handleAdd = async (e) => {
    e.preventDefault();
    if (!newFact.trim() || saving) return;
    setSaving(true);
    try {
      await onCreateMemory(newFact.trim());
      setNewFact('');
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-content" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <div className="modal-title">🧠 Persistent User Memory</div>
          <button className="modal-close-btn" onClick={onClose}>✕</button>
        </div>

        <div className="modal-body">
          <p style={{ fontSize: '12px', color: '#888', lineHeight: 1.4 }}>
            These persistent facts are stored across conversations. When the <strong>Memory toggle</strong> is ON,
            these facts are injected into the prompt. The AI can also autonomously add or remove facts using tools.
          </p>

          <form onSubmit={handleAdd} style={{ display: 'flex', gap: '8px' }}>
            <input
              type="text"
              className="input-text"
              placeholder="Add a new fact (e.g. 'I develop in Python and Rust')"
              value={newFact}
              onChange={(e) => setNewFact(e.target.value)}
            />
            <button
              type="submit"
              className="btn-sm-primary"
              disabled={!newFact.trim() || saving}
              style={{ whiteSpace: 'nowrap' }}
            >
              Add Fact
            </button>
          </form>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '8px', marginTop: '8px' }}>
            {memories.length === 0 ? (
              <div style={{ padding: '24px', textAlign: 'center', color: '#666', fontSize: '13px' }}>
                No facts stored in persistent memory yet.
              </div>
            ) : (
              memories.map((m) => (
                <div key={m.id} className="memory-item">
                  <span style={{ color: '#ffffff' }}>{m.content}</span>
                  <button
                    className="btn-msg-action"
                    onClick={() => onDeleteMemory(m.id)}
                    title="Permanently remove fact"
                  >
                    ✕
                  </button>
                </div>
              ))
            )}
          </div>
        </div>

        <div className="modal-footer">
          <button className="btn-sm-secondary" onClick={onClose}>
            Close
          </button>
        </div>
      </div>
    </div>
  );
}
