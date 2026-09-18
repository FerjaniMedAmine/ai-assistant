import React from 'react';

export default function HeaderBar({
  conversation,
  ragEnabled,
  onToggleRag,
  memoryEnabled,
  onToggleMemory,
  thinkingLevel,
  onChangeThinkingLevel,
}) {
  return (
    <header className="header-bar">
      <div className="header-left">
        <span className="header-conv-title">
          {conversation ? conversation.title : 'Select or Start a Conversation'}
        </span>
        {conversation && (
          <span className="header-conv-badge">
            {conversation.mode.toUpperCase()} MODE
          </span>
        )}
      </div>

      <div className="header-controls">
        {/* RAG Toggle */}
        <div className="toggle-group" title="Toggle RAG document context injection for messages">
          <span>RAG</span>
          <label className="toggle-switch">
            <input
              type="checkbox"
              checked={ragEnabled}
              onChange={(e) => onToggleRag(e.target.checked)}
            />
            <span className="toggle-slider"></span>
          </label>
          <span style={{ fontSize: '11px', color: ragEnabled ? '#ffffff' : '#666666', width: '24px' }}>
            {ragEnabled ? 'ON' : 'OFF'}
          </span>
        </div>

        {/* Memory Toggle */}
        <div className="toggle-group" title="Toggle persistent cross-conversation user facts injection">
          <span>Memory</span>
          <label className="toggle-switch">
            <input
              type="checkbox"
              checked={memoryEnabled}
              onChange={(e) => onToggleMemory(e.target.checked)}
            />
            <span className="toggle-slider"></span>
          </label>
          <span style={{ fontSize: '11px', color: memoryEnabled ? '#ffffff' : '#666666', width: '24px' }}>
            {memoryEnabled ? 'ON' : 'OFF'}
          </span>
        </div>

        {/* Thinking Level Selector */}
        <div className="toggle-group" title="Select Gemini Thinking Deliberation Budget">
          <span>Thinking:</span>
          <div className="segmented-control">
            <button
              className={`segment-btn ${thinkingLevel === 'low' ? 'active' : ''}`}
              onClick={() => onChangeThinkingLevel('low')}
            >
              Low
            </button>
            <button
              className={`segment-btn ${thinkingLevel === 'med' ? 'active' : ''}`}
              onClick={() => onChangeThinkingLevel('med')}
            >
              Med
            </button>
            <button
              className={`segment-btn ${thinkingLevel === 'high' ? 'active' : ''}`}
              onClick={() => onChangeThinkingLevel('high')}
            >
              High
            </button>
          </div>
        </div>
      </div>
    </header>
  );
}
