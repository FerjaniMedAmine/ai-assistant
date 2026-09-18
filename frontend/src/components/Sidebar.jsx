import React, { useState } from 'react';

export default function Sidebar({
  conversations,
  activeConvId,
  activeConv,
  messagesCount = 0,
  onSelectConversation,
  onCreateConversation,
  onDeleteConversation,
  onSwitchMode,
  onOpenMemoryModal,
  onOpenKnowledgeModal,
  memoryCount = 0,
  ragPointsCount = 0,
}) {
  const currentMode = activeConv?.mode || 'linear';
  const isCurrentEmpty = messagesCount === 0;

  const handleCreate = () => {
    if (isCurrentEmpty) {
      return; // Already in an empty conversation, prevent spam
    }
    onCreateConversation('New Conversation', currentMode);
  };

  const handleModeChange = (mode) => {
    if (onSwitchMode) {
      onSwitchMode(mode);
    }
  };

  return (
    <aside className="sidebar">
      <div className="sidebar-header">
        <div className="brand-title">
          <span>AI ASSISTANT</span>
        </div>

        <div className="new-chat-container">
          <button
            className="btn-new-chat"
            onClick={handleCreate}
            disabled={isCurrentEmpty}
            style={{
              opacity: isCurrentEmpty ? 0.5 : 1,
              cursor: isCurrentEmpty ? 'not-allowed' : 'pointer'
            }}
            title={isCurrentEmpty ? 'Current chat is already empty' : 'Start a new conversation'}
          >
            + New Chat
          </button>
        </div>

        {/* Mode switcher - switching opens a conversation in that mode */}
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', fontSize: '11px', color: '#888' }}>
          <span>Mode:</span>
          <div className="segmented-control">
            <button
              className={`segment-btn ${currentMode === 'linear' ? 'active' : ''}`}
              onClick={() => handleModeChange('linear')}
              title="Switch to Linear Chat thread"
            >
              Linear
            </button>
            <button
              className={`segment-btn ${currentMode === 'isolated' ? 'active' : ''}`}
              onClick={() => handleModeChange('isolated')}
              title="Switch to Isolated Questions mode"
            >
              Isolated
            </button>
          </div>
        </div>
      </div>

      <div className="sidebar-conversations">
        {conversations.length === 0 ? (
          <div style={{ padding: '20px 12px', fontSize: '12px', color: '#555', textAlign: 'center' }}>
            No conversations yet. Start a new chat above.
          </div>
        ) : (
          conversations.map((conv) => {
            const isActive = conv.id === activeConvId;
            return (
              <div
                key={conv.id}
                className={`conversation-item ${isActive ? 'active' : ''}`}
                onClick={() => onSelectConversation(conv.id)}
              >
                <div className="conv-info">
                  <span className="conv-title">{conv.title || 'Untitled'}</span>
                  <div className="conv-meta">
                    <span className="mode-tag">{conv.mode}</span>
                    <span>{conv.message_count ?? 0} msgs</span>
                  </div>
                </div>

                <button
                  className="btn-icon-delete"
                  title="Hard delete conversation"
                  onClick={(e) => {
                    e.stopPropagation();
                    if (window.confirm(`Delete conversation "${conv.title}" permanently?`)) {
                      onDeleteConversation(conv.id);
                    }
                  }}
                >
                  ✕
                </button>
              </div>
            );
          })
        )}
      </div>

      <div className="sidebar-footer">
        <button className="sidebar-footer-btn" onClick={onOpenMemoryModal}>
          <span>🧠 Persistent Memory</span>
          <span style={{ fontFamily: 'var(--font-mono)', fontSize: '11px' }}>
            {memoryCount} facts
          </span>
        </button>

        <button className="sidebar-footer-btn" onClick={onOpenKnowledgeModal}>
          <span>📚 RAG Knowledge Base</span>
          <span style={{ fontFamily: 'var(--font-mono)', fontSize: '11px' }}>
            {ragPointsCount} chunks
          </span>
        </button>
      </div>
    </aside>
  );
}
