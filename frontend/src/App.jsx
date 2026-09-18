import React, { useState, useEffect } from 'react';
import {
  getConversations,
  createConversation,
  getConversation,
  deleteConversation,
  sendMessage,
  streamMessage,
  editMessage,
  deleteMessage,
  getMemories,
  createMemory,
  deleteMemory,
  getDocumentStats,
  ingestDocument,
  clearDocuments,
} from './api/client';
import { useLocalStorage } from './hooks/useLocalStorage';
import Sidebar from './components/Sidebar';
import HeaderBar from './components/HeaderBar';
import LinearChatView from './components/LinearChatView';
import IsolatedChatView from './components/IsolatedChatView';
import MessageInput from './components/MessageInput';
import MemoryModal from './components/MemoryModal';
import KnowledgeModal from './components/KnowledgeModal';

export default function App() {
  // LocalStorage persistent state
  const [activeConvId, setActiveConvId] = useLocalStorage('ai_active_conv_id', null);
  const [ragEnabled, setRagEnabled] = useLocalStorage('ai_rag_enabled', false);
  const [memoryEnabled, setMemoryEnabled] = useLocalStorage('ai_memory_enabled', false);
  const [thinkingLevel, setThinkingLevel] = useLocalStorage('ai_thinking_level', 'med');
  const [drafts, setDrafts] = useLocalStorage('ai_drafts', {});

  // Runtime application state
  const [conversations, setConversations] = useState([]);
  const [activeConv, setActiveConv] = useState(null);
  const [messages, setMessages] = useState([]);
  const [memories, setMemories] = useState([]);
  const [docStats, setDocStats] = useState({ total_points: 0 });
  const [loading, setLoading] = useState(false);
  const [errorMsg, setErrorMsg] = useState(null);

  // Latest interaction metadata
  const [latestRagSources, setLatestRagSources] = useState([]);
  const [latestToolCalls, setLatestToolCalls] = useState([]);
  const [latestThoughts, setLatestThoughts] = useState(null);

  // Modal visibility
  const [isMemoryOpen, setIsMemoryOpen] = useState(false);
  const [isKnowledgeOpen, setIsKnowledgeOpen] = useState(false);

  const [draftMode, setDraftMode] = useLocalStorage('ai_draft_mode', 'linear');

  // Current draft for active conversation
  const currentDraftKey = activeConvId || 'new_draft';
  const currentDraft = drafts[currentDraftKey] || '';

  const handleDraftChange = (newText) => {
    setDrafts((prev) => ({
      ...prev,
      [currentDraftKey]: newText,
    }));
  };

  // Load initial data
  const loadInitialData = async () => {
    try {
      const [convs, mems, stats] = await Promise.all([
        getConversations(),
        getMemories(),
        getDocumentStats(),
      ]);
      setConversations(convs);
      setMemories(mems);
      setDocStats(stats);

      // Select active conversation if it exists in conversations with messages
      if (activeConvId && convs.some((c) => c.id === activeConvId)) {
        loadMessages(activeConvId);
      } else {
        // Start on a clean fresh chat draft screen
        setActiveConvId(null);
        setActiveConv({ id: null, title: 'New Conversation', mode: draftMode });
        setMessages([]);
      }
    } catch (err) {
      console.error('Failed to load initial data:', err);
      setErrorMsg(err.message);
    }
  };

  const loadMessages = async (convId) => {
    try {
      const data = await getConversation(convId);
      setActiveConv(data);
      setMessages(data.messages || []);
      setLatestRagSources([]);
      setLatestToolCalls([]);
      setLatestThoughts(null);
    } catch (err) {
      console.error('Failed to load messages:', err);
      setErrorMsg(err.message);
    }
  };

  useEffect(() => {
    loadInitialData();
  }, []);

  const handleSelectConversation = (id) => {
    setActiveConvId(id);
    loadMessages(id);
  };

  // Start fresh chat draft (zero database writes until message sent)
  const handleStartNewChat = (mode = draftMode) => {
    if (activeConvId === null && messages.length === 0 && draftMode === mode) {
      return; // Already on fresh chat in this mode
    }
    setDraftMode(mode);
    setActiveConvId(null);
    setActiveConv({ id: null, title: 'New Conversation', mode: mode });
    setMessages([]);
    setLatestRagSources([]);
    setLatestToolCalls([]);
    setLatestThoughts(null);
  };

  const handleSwitchMode = (newMode) => {
    if (activeConv?.mode === newMode && activeConvId === null) {
      return;
    }
    setDraftMode(newMode);
    // If currently viewing an existing conversation, open a fresh chat in the new mode
    if (activeConvId !== null) {
      handleStartNewChat(newMode);
    } else {
      setActiveConv((prev) => ({ ...prev, mode: newMode }));
    }
  };

  const handleDeleteConversation = async (id) => {
    try {
      await deleteConversation(id);
      const remaining = conversations.filter((c) => c.id !== id);
      setConversations(remaining);

      // Clean draft for deleted conversation
      setDrafts((prev) => {
        const copy = { ...prev };
        delete copy[id];
        return copy;
      });

      if (remaining.length > 0) {
        const nextId = remaining[0].id;
        setActiveConvId(nextId);
        loadMessages(nextId);
      } else {
        handleStartNewChat(draftMode);
      }
    } catch (err) {
      setErrorMsg(`Failed to delete conversation: ${err.message}`);
    }
  };

  const handleSendMessage = async ({ content, linked_message_id }) => {
    setLoading(true);
    setErrorMsg(null);

    const tempUserId = 'temp-user-' + Date.now();
    const tempAsstId = 'temp-asst-' + Date.now();

    const optimisticUserMsg = {
      id: tempUserId,
      role: 'user',
      content,
      created_at: new Date().toISOString(),
      rag_enabled: ragEnabled,
      memory_enabled: memoryEnabled,
      thinking_level: thinkingLevel,
      linked_message_id,
    };

    const streamingAsstMsg = {
      id: tempAsstId,
      role: 'assistant',
      content: '',
      created_at: new Date().toISOString(),
      rag_enabled: ragEnabled,
      memory_enabled: memoryEnabled,
      thinking_level: thinkingLevel,
      isStreaming: true,
    };

    // Immediately show both user message and assistant placeholder in chat view
    setMessages((prev) => [...prev, optimisticUserMsg, streamingAsstMsg]);
    handleDraftChange('');

    try {
      let targetConvId = activeConvId;

      // If currently on a fresh draft chat, persist the conversation to PostgreSQL on first message!
      if (!targetConvId) {
        const newConv = await createConversation('New Conversation', draftMode);
        targetConvId = newConv.id;
        setActiveConvId(targetConvId);
        setActiveConv(newConv);
      }

      await streamMessage(
        targetConvId,
        {
          content,
          rag_enabled: ragEnabled,
          memory_enabled: memoryEnabled,
          thinking_level: thinkingLevel,
          linked_message_id,
        },
        {
          onToken: (token) => {
            setMessages((prev) =>
              prev.map((m) =>
                m.id === tempAsstId
                  ? { ...m, content: m.content + token, isStreaming: true }
                  : m
              )
            );
          },
          onToolCall: (tc) => {
            setLatestToolCalls((prev) => [...prev, tc]);
          },
          onDone: async (reply) => {
            // Replace temporary messages with finalized database records
            setMessages((prev) =>
              prev.map((m) => {
                if (m.id === tempUserId) return reply.user_message;
                if (m.id === tempAsstId) return reply.assistant_message;
                return m;
              })
            );
            setLatestRagSources(reply.rag_sources || []);
            setLatestToolCalls(reply.tool_calls || []);

            // Refresh conversations list: now that it has messages, it appears in the sidebar!
            const updatedConvs = await getConversations();
            setConversations(updatedConvs);
            const cur = updatedConvs.find((c) => c.id === targetConvId);
            if (cur) setActiveConv((prev) => ({ ...prev, title: cur.title, mode: cur.mode }));

            // If tool calls included memory modifications, refresh memories
            const hasMemoryChange = (reply.tool_calls || []).some(
              (tc) => tc.tool === 'add_to_memory' || tc.tool === 'remove_from_memory'
            );
            if (hasMemoryChange) {
              const updatedMems = await getMemories();
              setMemories(updatedMems);
            }
            setLoading(false);
          },
          onError: (err) => {
            setErrorMsg(`Failed to stream message: ${err.message}`);
            // Remove empty assistant placeholder if failed completely
            setMessages((prev) =>
              prev.filter((m) => !(m.id === tempAsstId && !m.content))
            );
            setLoading(false);
          },
        }
      );
    } catch (err) {
      setErrorMsg(`Failed to send message: ${err.message}`);
      setMessages((prev) =>
        prev.filter((m) => !(m.id === tempAsstId && !m.content))
      );
      setLoading(false);
    }
  };

  const handleEditMessage = async (messageId, newContent) => {
    if (!activeConvId) return;
    setLoading(true);
    setErrorMsg(null);

    try {
      await editMessage(messageId, newContent);
      // Reload the conversation messages to reflect DB state after hard deletion of subsequent messages
      await loadMessages(activeConvId);

      // Refresh memory list in case tool was executed
      const updatedMems = await getMemories();
      setMemories(updatedMems);
    } catch (err) {
      setErrorMsg(`Failed to edit message: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  const handleDeleteMessage = async (messageId) => {
    try {
      await deleteMessage(messageId);
      setMessages((prev) => prev.filter((m) => m.id !== messageId));
    } catch (err) {
      setErrorMsg(`Failed to delete message: ${err.message}`);
    }
  };

  const handleCreateMemory = async (content) => {
    const mem = await createMemory(content);
    setMemories((prev) => [mem, ...prev]);
  };

  const handleDeleteMemory = async (id) => {
    await deleteMemory(id);
    setMemories((prev) => prev.filter((m) => m.id !== id));
  };

  const handleIngestDocument = async (formData) => {
    const res = await ingestDocument(formData);
    const stats = await getDocumentStats();
    setDocStats(stats);
    return res;
  };

  const handleClearAllDocuments = async () => {
    await clearDocuments();
    const stats = await getDocumentStats();
    setDocStats(stats);
  };

  const isIsolatedMode = activeConv?.mode === 'isolated';

  return (
    <div className="app-container">
      {/* Left Sidebar */}
      <Sidebar
        conversations={conversations}
        activeConvId={activeConvId}
        activeConv={activeConv}
        messagesCount={activeConvId ? messages.length : 0}
        onSelectConversation={handleSelectConversation}
        onCreateConversation={() => handleStartNewChat(draftMode)}
        onDeleteConversation={handleDeleteConversation}
        onSwitchMode={handleSwitchMode}
        onOpenMemoryModal={() => setIsMemoryOpen(true)}
        onOpenKnowledgeModal={() => setIsKnowledgeOpen(true)}
        memoryCount={memories.length}
        ragPointsCount={docStats.total_points ?? 0}
      />

      {/* Main Chat Workspace */}
      <main className="main-area">
        <HeaderBar
          conversation={activeConv}
          ragEnabled={ragEnabled}
          onToggleRag={setRagEnabled}
          memoryEnabled={memoryEnabled}
          onToggleMemory={setMemoryEnabled}
          thinkingLevel={thinkingLevel}
          onChangeThinkingLevel={setThinkingLevel}
        />

        {errorMsg && (
          <div
            style={{
              padding: '10px 24px',
              backgroundColor: '#2a1414',
              borderBottom: '1px solid #ff4444',
              color: '#ff8888',
              fontSize: '13px',
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
            }}
          >
            <span>{errorMsg}</span>
            <button
              onClick={() => setErrorMsg(null)}
              style={{ background: 'none', border: 'none', color: '#ff8888', cursor: 'pointer' }}
            >
              ✕
            </button>
          </div>
        )}

        <div className="chat-scroll-area">
          {isIsolatedMode ? (
            <IsolatedChatView
              messages={messages}
              onEditMessage={handleEditMessage}
              onDeleteMessage={handleDeleteMessage}
              loading={loading}
              latestRagSources={latestRagSources}
              latestToolCalls={latestToolCalls}
              latestThoughts={latestThoughts}
            />
          ) : (
            <LinearChatView
              messages={messages}
              onEditMessage={handleEditMessage}
              onDeleteMessage={handleDeleteMessage}
              loading={loading}
              latestRagSources={latestRagSources}
              latestToolCalls={latestToolCalls}
              latestThoughts={latestThoughts}
            />
          )}
        </div>

        {/* Message Composer & Isolated Exchange Picker */}
        <MessageInput
          mode={activeConv?.mode || 'linear'}
          messages={messages}
          onSendMessage={handleSendMessage}
          loading={loading}
          draft={currentDraft}
          onDraftChange={handleDraftChange}
        />
      </main>

      {/* Persistent Memory Manager Modal */}
      <MemoryModal
        isOpen={isMemoryOpen}
        onClose={() => setIsMemoryOpen(false)}
        memories={memories}
        onCreateMemory={handleCreateMemory}
        onDeleteMemory={handleDeleteMemory}
      />

      {/* RAG Knowledge Base Ingestion Modal */}
      <KnowledgeModal
        isOpen={isKnowledgeOpen}
        onClose={() => setIsKnowledgeOpen(false)}
        stats={docStats}
        onIngest={handleIngestDocument}
        onClearAll={handleClearAllDocuments}
      />
    </div>
  );
}
