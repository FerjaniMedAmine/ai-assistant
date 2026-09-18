import React, { useState } from 'react';

export default function IsolatedChatView({
  messages,
  onEditMessage,
  onDeleteMessage,
  loading,
  latestRagSources = [],
  latestToolCalls = [],
  latestThoughts = null,
}) {
  const [editingMsgId, setEditingMsgId] = useState(null);
  const [editContent, setEditContent] = useState('');
  const [expandedThoughtsId, setExpandedThoughtsId] = useState(null);

  // Group messages into exchange pairs (hk, ak)
  const exchanges = [];
  let currentPair = null;

  for (let i = 0; i < messages.length; i++) {
    const msg = messages[i];
    if (msg.role === 'user') {
      currentPair = { user: msg, assistant: null, index: exchanges.length + 1 };
      exchanges.push(currentPair);
    } else if (msg.role === 'assistant' && currentPair) {
      currentPair.assistant = msg;
    }
  }

  const startEditing = (userMsg) => {
    setEditingMsgId(userMsg.id);
    setEditContent(userMsg.content);
  };

  const cancelEditing = () => {
    setEditingMsgId(null);
    setEditContent('');
  };

  const saveEdit = (msgId) => {
    if (!editContent.trim()) return;
    onEditMessage(msgId, editContent.trim());
    setEditingMsgId(null);
  };

  return (
    <div className="isolated-container">
      <div className="isolated-banner">
        <div>
          <div className="isolated-banner-title">Isolated Questions Mode</div>
          <div className="isolated-banner-desc">
            Each question is standalone by default with zero automatic history.
            You can selectively link a new question to any previous exchange using the picker below.
          </div>
        </div>
        <div style={{ fontFamily: 'var(--font-mono)', fontSize: '11px', color: '#888' }}>
          {exchanges.length} exchange(s)
        </div>
      </div>

      <div className="isolated-exchanges-grid">
        {exchanges.length === 0 ? (
          <div className="empty-state">
            <div className="empty-state-title">No Exchanges Yet</div>
            <div className="empty-state-subtitle">
              Ask your first isolated question below. When drafting future questions,
              you can attach any prior exchange as reference context.
            </div>
          </div>
        ) : (
          exchanges.map((ex) => {
            const isEditing = editingMsgId === ex.user?.id;
            const isLatest = ex.index === exchanges.length;

            // Find linked exchange info if linked_message_id is set
            let linkedExchangeIndex = null;
            if (ex.user?.linked_message_id) {
              const linkedTarget = exchanges.find(
                (target) => target.user?.id === ex.user.linked_message_id
              );
              if (linkedTarget) {
                linkedExchangeIndex = linkedTarget.index;
              }
            }

            return (
              <div key={ex.user.id} className="exchange-card">
                <div className="exchange-header">
                  <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                    <span className="exchange-index">EXCHANGE #{ex.index}</span>
                    {linkedExchangeIndex && (
                      <span className="exchange-link-badge">
                        🔗 Linked to Exchange #{linkedExchangeIndex}
                      </span>
                    )}
                  </div>

                  <div style={{ display: 'flex', gap: '6px' }}>
                    <button
                      className="btn-msg-action"
                      onClick={() => startEditing(ex.user)}
                      title="Edit question & regenerate this exchange"
                    >
                      Edit
                    </button>
                    <button
                      className="btn-msg-action"
                      onClick={() => {
                        if (window.confirm('Permanently delete this exchange?')) {
                          onDeleteMessage(ex.user.id);
                        }
                      }}
                      title="Permanently hard-delete exchange"
                    >
                      Delete
                    </button>
                  </div>
                </div>

                <div className="exchange-body">
                  {/* User Question */}
                  <div className="exchange-q">
                    <div className="exchange-q-label">Question</div>
                    {isEditing ? (
                      <div className="edit-message-box">
                        <textarea
                          className="edit-message-textarea"
                          value={editContent}
                          onChange={(e) => setEditContent(e.target.value)}
                          autoFocus
                        />
                        <div className="edit-actions">
                          <button className="btn-sm-secondary" onClick={cancelEditing}>
                            Cancel
                          </button>
                          <button
                            className="btn-sm-primary"
                            onClick={() => saveEdit(ex.user.id)}
                          >
                            Save & Regenerate
                          </button>
                        </div>
                      </div>
                    ) : (
                      <div className="exchange-q-text">{ex.user.content}</div>
                    )}
                  </div>

                  {/* Assistant Answer */}
                  <div className="exchange-a">
                    <div className="exchange-a-label">Assistant Response</div>
                    {ex.assistant ? (
                      <div className="exchange-a-text">
                        {ex.assistant.content || (
                          <span style={{ color: '#888', fontStyle: 'italic', fontSize: '13px' }}>
                            Deliberating...
                          </span>
                        )}
                        {ex.assistant.isStreaming && <span style={{ opacity: 0.7, marginLeft: '2px' }}>▊</span>}
                      </div>
                    ) : (
                      <div style={{ color: '#666', fontStyle: 'italic', fontSize: '13px' }}>
                        Waiting for response...
                      </div>
                    )}
                  </div>
                </div>
              </div>
            );
          })
        )}
      </div>

      {loading && !messages.some((m) => m.isStreaming || (m.role === 'assistant' && !m.content)) && (
        <div className="exchange-card" style={{ padding: '16px', color: '#888' }}>
          <span>Deliberating response for isolated question...</span>
        </div>
      )}
    </div>
  );
}
