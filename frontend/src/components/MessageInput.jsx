import React, { useState, useEffect, useRef } from 'react';

export default function MessageInput({
  mode,
  messages,
  onSendMessage,
  loading,
  draft,
  onDraftChange,
}) {
  const [selectedExchangeId, setSelectedExchangeId] = useState('');
  const textareaRef = useRef(null);

  // Group messages for the exchange picker in isolated mode
  const priorExchanges = [];
  if (mode === 'isolated') {
    let currentPair = null;
    for (const m of messages) {
      if (m.role === 'user') {
        currentPair = { user: m, assistant: null, index: priorExchanges.length + 1 };
        priorExchanges.push(currentPair);
      } else if (m.role === 'assistant' && currentPair) {
        currentPair.assistant = m;
      }
    }
  }

  // Adjust textarea height dynamically
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 180)}px`;
    }
  }, [draft]);

  const handleSubmit = (e) => {
    e?.preventDefault();
    if (!draft.trim() || loading) return;

    onSendMessage({
      content: draft.trim(),
      linked_message_id: mode === 'isolated' && selectedExchangeId ? selectedExchangeId : null,
    });

    onDraftChange('');
    setSelectedExchangeId('');
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  return (
    <div className="composer-area">
      {/* If Isolated mode, show the Prior Exchange Reference Selector */}
      {mode === 'isolated' && (
        <div className="exchange-picker-bar">
          <div className="exchange-picker-label">
            <span>🔗 Attach Prior Exchange Context:</span>
          </div>

          <select
            className="exchange-select"
            value={selectedExchangeId}
            onChange={(e) => setSelectedExchangeId(e.target.value)}
            disabled={priorExchanges.length === 0}
          >
            <option value="">None (Independent Question)</option>
            {priorExchanges.map((ex) => {
              const snippet =
                ex.user.content.length > 50
                  ? ex.user.content.substring(0, 50) + '...'
                  : ex.user.content;
              return (
                <option key={ex.user.id} value={ex.user.id}>
                  Exchange #{ex.index}: "{snippet}"
                </option>
              );
            })}
          </select>
        </div>
      )}

      <div className={`composer-box ${mode === 'isolated' ? 'has-picker' : ''}`}>
        <textarea
          ref={textareaRef}
          className="composer-textarea"
          placeholder="Type a message..."
          value={draft}
          onChange={(e) => onDraftChange(e.target.value)}
          onKeyDown={handleKeyDown}
          disabled={loading}
          rows={1}
        />

        <div className="composer-footer">
          <span className="composer-hints">
            Press <strong>Enter</strong> to send &bull; <strong>Shift + Enter</strong> for newline
          </span>
          <button
            className="btn-send"
            onClick={handleSubmit}
            disabled={!draft.trim() || loading}
          >
            {loading ? 'Sending...' : 'Send Message'}
          </button>
        </div>
      </div>
    </div>
  );
}
