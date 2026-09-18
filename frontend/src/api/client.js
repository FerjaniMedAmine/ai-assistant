import axios from 'axios';

// Centralized Axios instance
const apiClient = axios.create({
  baseURL: '/api',
  headers: {
    'Content-Type': 'application/json',
  },
  timeout: 120000, // 2 minutes timeout for complex RAG/thinking deliberation
});

// Response interceptor for consistent error messaging
apiClient.interceptors.response.use(
  (response) => response,
  (error) => {
    const message =
      error.response?.data?.detail ||
      error.message ||
      'An unexpected error occurred communicating with the server.';
    return Promise.reject(new Error(message));
  }
);

// Conversations API
export const getConversations = async () => {
  const response = await apiClient.get('/conversations');
  return response.data;
};

export const createConversation = async (title = 'New Conversation', mode = 'linear') => {
  const response = await apiClient.post('/conversations', { title, mode });
  return response.data;
};

export const getConversation = async (conversationId) => {
  const response = await apiClient.get(`/conversations/${conversationId}`);
  return response.data;
};

export const deleteConversation = async (conversationId) => {
  const response = await apiClient.delete(`/conversations/${conversationId}`);
  return response.data;
};

// Messages API
export const sendMessage = async (conversationId, payload) => {
  const response = await apiClient.post(`/conversations/${conversationId}/messages`, {
    content: payload.content,
    rag_enabled: payload.rag_enabled ?? false,
    memory_enabled: payload.memory_enabled ?? false,
    thinking_level: payload.thinking_level ?? 'med',
    linked_message_id: payload.linked_message_id ?? null,
  });
  return response.data;
};

export const streamMessage = async (conversationId, payload, { onToken, onToolCall, onDone, onError }) => {
  try {
    const response = await fetch(`/api/conversations/${conversationId}/messages/stream`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        content: payload.content,
        rag_enabled: payload.rag_enabled ?? false,
        memory_enabled: payload.memory_enabled ?? false,
        thinking_level: payload.thinking_level ?? 'med',
        linked_message_id: payload.linked_message_id ?? null,
      }),
    });

    if (!response.ok) {
      let errText = 'Streaming request failed';
      try {
        const errJson = await response.json();
        errText = errJson.detail || errText;
      } catch (e) {
        errText = await response.text();
      }
      throw new Error(errText);
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder('utf-8');
    let buffer = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop() || '';

      for (const line of lines) {
        const trimmed = line.trim();
        if (trimmed.startsWith('data: ')) {
          try {
            const data = JSON.parse(trimmed.slice(6));
            if (data.type === 'token') {
              if (onToken) onToken(data.token);
            } else if (data.type === 'tool_call') {
              if (onToolCall) onToolCall(data);
            } else if (data.type === 'done') {
              if (onDone) onDone(data);
            } else if (data.type === 'error') {
              if (onError) onError(new Error(data.error));
            }
          } catch (e) {
            console.error('Failed to parse SSE JSON chunk:', trimmed, e);
          }
        }
      }
    }
  } catch (err) {
    if (onError) onError(err);
    else throw err;
  }
};

export const editMessage = async (messageId, content) => {
  const response = await apiClient.put(`/messages/${messageId}`, { content });
  return response.data;
};

export const deleteMessage = async (messageId) => {
  const response = await apiClient.delete(`/messages/${messageId}`);
  return response.data;
};

// Memory API
export const getMemories = async () => {
  const response = await apiClient.get('/memory');
  return response.data;
};

export const createMemory = async (content) => {
  const response = await apiClient.post('/memory', { content });
  return response.data;
};

export const deleteMemory = async (memoryId) => {
  const response = await apiClient.delete(`/memory/${memoryId}`);
  return response.data;
};

// RAG Documents API
export const getDocumentStats = async () => {
  const response = await apiClient.get('/documents');
  return response.data;
};

export const ingestDocument = async (formData) => {
  const response = await apiClient.post('/documents/ingest', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  });
  return response.data;
};

export const clearDocuments = async () => {
  const response = await apiClient.delete('/documents');
  return response.data;
};

// Config API
export const getRuntimeConfig = async () => {
  const response = await apiClient.get('/config');
  return response.data;
};

export default apiClient;
