import React, { useState } from 'react';

export default function LinearChatView({
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
  const [expandedThoughts, setExpandedThoughts] = useState(false);
  const [expandedSources, setExpandedSources] = useState(false);

  const startEditing = (msg) => {
    setEditingMsgId(msg.id);
    setEditContent(msg.content);
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
    <div className="linear-thread">
      {messages.length === 0 ? (
        <div className="empty-state">
          <div className="empty-state-title">Linear Conversation</div>
          <div className="empty-state-subtitle">
            Every message is part of an ongoing chat thread with full conversation history preserved.
            Type your message below to begin.
          </div>
        </div>
      ) : (
        messages.map((msg, index) => {
          const isUser = msg.role === 'user';
          const isEditing = editingMsgId === msg.id;
          const isLatestAsst = !isUser && index === messages.length - 1;

          return (
            <div key={msg.id} className="message-row">
              <div className="message-meta-header">
                <span className={`message-author ${msg.role}`}>
                  {isUser ? 'You' : 'Assistant'}
                </span>
                <div className="message-actions">
                  {isUser && (
                    <button
                      className="btn-msg-action"
                      onClick={() => startEditing(msg)}
                      title="Edit this message and regenerate AI reply"
                    >
                      Edit
                    </button>
                  )}
                  <button
                    className="btn-msg-action"
                    onClick={() => {
                      if (window.confirm('Delete this message permanently from PostgreSQL?')) {
                        onDeleteMessage(msg.id);
                      }
                    }}
                    title="Permanently hard-delete message"
                  >
                    Delete
                  </button>
                </div>
              </div>

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
                    <button className="btn-sm-primary" onClick={() => saveEdit(msg.id)}>
                      Save & Regenerate
                    </button>
                  </div>
                </div>
              ) : (
                <div className={`message-bubble ${msg.role}`}>
                  <div style={{ whiteSpace: 'pre-wrap' }}>
                    {msg.content || (
                      <span style={{ color: '#888', fontStyle: 'italic', fontSize: '13px' }}>
                        Deliberating...
                      </span>
                    )}
                    {msg.isStreaming && <span style={{ opacity: 0.7, marginLeft: '2px' }}>▊</span>}
                  </div>
                </div>
              )}
            </div>
          );
        })
      )}

      {loading && !messages.some((m) => m.isStreaming || (m.role === 'assistant' && !m.content)) && (
        <div className="message-row">
          <div className="message-meta-header">
            <span className="message-author assistant">Assistant</span>
          </div>
          <div className="message-bubble assistant">
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px', color: '#888' }}>
              <span style={{ animation: 'pulse 1s infinite' }}>●</span>
              <span>Deliberating response...</span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
