import React, { useState } from 'react';

export default function KnowledgeModal({
  isOpen,
  onClose,
  stats,
  onIngest,
  onClearAll,
}) {
  const [selectedFile, setSelectedFile] = useState(null);
  const [rawText, setRawText] = useState('');
  const [sourceName, setSourceName] = useState('');
  const [ingesting, setIngesting] = useState(false);
  const [statusMsg, setStatusMsg] = useState('');

  if (!isOpen) return null;

  const handleIngestFile = async () => {
    if (!selectedFile) return;
    setIngesting(true);
    setStatusMsg('Chunking and embedding document with BGE-M3 (dense + sparse)...');

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      const res = await onIngest(formData);
      setStatusMsg(`Successfully indexed ${res.chunks_indexed} chunks for "${res.source_name}".`);
      setSelectedFile(null);
    } catch (err) {
      setStatusMsg(`Ingestion failed: ${err.message}`);
    } finally {
      setIngesting(false);
    }
  };

  const handleIngestText = async () => {
    if (!rawText.trim()) return;
    setIngesting(true);
    setStatusMsg('Chunking and embedding text with BGE-M3...');

    const formData = new FormData();
    formData.append('text', rawText.trim());
    formData.append('source_name', sourceName.trim() || 'user_note');

    try {
      const res = await onIngest(formData);
      setStatusMsg(`Successfully indexed ${res.chunks_indexed} chunks for "${res.source_name}".`);
      setRawText('');
      setSourceName('');
    } catch (err) {
      setStatusMsg(`Ingestion failed: ${err.message}`);
    } finally {
      setIngesting(false);
    }
  };

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-content" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <div className="modal-title">📚 RAG Knowledge Base (Qdrant & BGE-M3)</div>
          <button className="modal-close-btn" onClick={onClose}>✕</button>
        </div>

        <div className="modal-body">
          <div style={{ display: 'flex', justifyContent: 'space-between', padding: '10px 14px', background: '#161616', borderRadius: '6px', border: '1px solid #222', fontSize: '12px' }}>
            <div>
              <span style={{ color: '#888' }}>Collection: </span>
              <strong style={{ color: '#fff' }}>{stats?.collection_name || 'assistant_knowledge'}</strong>
            </div>
            <div>
              <span style={{ color: '#888' }}>Total Indexed Chunks: </span>
              <strong style={{ color: '#fff' }}>{stats?.total_points ?? 0}</strong>
            </div>
          </div>

          {statusMsg && (
            <div style={{ padding: '8px 12px', background: '#1c1c1c', border: '1px solid #333', borderRadius: '4px', fontSize: '12px', color: '#eaeaea' }}>
              {statusMsg}
            </div>
          )}

          {/* Upload File */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
            <span style={{ fontSize: '12px', fontWeight: 600, color: '#e0e0e0' }}>Option 1: Upload Document</span>
            <div style={{ display: 'flex', gap: '8px' }}>
              <input
                type="file"
                accept=".txt,.md,.pdf,.json,.py,.csv"
                onChange={(e) => setSelectedFile(e.target.files[0] || null)}
                style={{ flex: 1, fontSize: '12px', color: '#888' }}
              />
              <button
                className="btn-sm-primary"
                onClick={handleIngestFile}
                disabled={!selectedFile || ingesting}
              >
                {ingesting ? 'Indexing...' : 'Index File'}
              </button>
            </div>
          </div>

          <div style={{ borderBottom: '1px solid #222', margin: '4px 0' }}></div>

          {/* Paste Raw Text */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
            <span style={{ fontSize: '12px', fontWeight: 600, color: '#e0e0e0' }}>Option 2: Paste Raw Text</span>
            <input
              type="text"
              className="input-text"
              placeholder="Source Title (e.g. 'Project Guidelines')"
              value={sourceName}
              onChange={(e) => setSourceName(e.target.value)}
            />
            <textarea
              className="edit-message-textarea"
              placeholder="Paste content here..."
              rows={4}
              value={rawText}
              onChange={(e) => setRawText(e.target.value)}
            />
            <button
              className="btn-sm-primary"
              onClick={handleIngestText}
              disabled={!rawText.trim() || ingesting}
              style={{ alignSelf: 'flex-end' }}
            >
              {ingesting ? 'Indexing...' : 'Index Text'}
            </button>
          </div>
        </div>

        <div className="modal-footer" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <button
            className="btn-msg-action"
            style={{ color: '#ff6666', borderColor: '#552222', padding: '6px 12px', fontSize: '11px' }}
            onClick={async () => {
              if (window.confirm('Are you sure you want to permanently purge all documents and vector points from Qdrant?')) {
                if (onClearAll) {
                  await onClearAll();
                  setStatusMsg('All vector points have been purged from Qdrant.');
                }
              }
            }}
          >
            🗑️ Clear All Knowledge
          </button>
          <button className="btn-sm-secondary" onClick={onClose}>
            Close
          </button>
        </div>
      </div>
    </div>
  );
}
