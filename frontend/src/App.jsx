import { useState, useRef, useEffect } from 'react';
import { Send, Loader2, AlertCircle, Bot, User, BookOpen } from 'lucide-react';

function App() {
  const [history, setHistory] = useState([]);
  const [question, setQuestion] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [history, isLoading]);

  const handleSubmit = async (e) => {
    if (e) e.preventDefault();
    if (!question.trim() || isLoading) return;

    const currentQuestion = question.trim();
    setQuestion('');
    setError(null);
    setHistory(prev => [...prev, { role: 'user', text: currentQuestion }]);
    setIsLoading(true);

    try {
      const response = await fetch('http://127.0.0.1:8000/ask', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ question: currentQuestion }),
      });

      if (!response.ok) {
        throw new Error(`Server responded with ${response.status}`);
      }

      const data = await response.json();
      setHistory(prev => [...prev, { 
        role: 'assistant', 
        text: data.answer, 
        sources: data.sources 
      }]);
    } catch (err) {
      const errorMessage = 'Failed to connect to the medical assistant. Please ensure the backend is running at http://127.0.0.1:8000.';
      setError(errorMessage);
      setHistory(prev => [...prev, { role: 'error', text: 'Error: ' + err.message }]);
      setQuestion(currentQuestion); // restore question so user doesn't lose it
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  return (
    <div className="app-container">
      <header className="app-header">
        <div className="header-content">
          <h1>MedRAG <span className="subtitle">— AI Medical Assistant</span></h1>
          <div className="disclaimer">
            <AlertCircle size={14} />
            <span>Generated from a medical knowledge base, not a substitute for professional medical advice.</span>
          </div>
        </div>
      </header>

      <main className="chat-container">
        <div className="messages-area">
          {history.length === 0 && (
            <div className="empty-state">
              <Bot size={48} className="empty-icon" />
              <h2>How can I help you today?</h2>
              <p>Ask a medical question and I will consult the knowledge base for an answer.</p>
            </div>
          )}
          
          {history.map((msg, idx) => {
            if (msg.role === 'error') {
              return (
                <div key={idx} className="error-message">
                  <AlertCircle size={18} />
                  {msg.text}
                </div>
              );
            }

            return (
              <div key={idx} className={`message-wrapper ${msg.role}`}>
                <div className="message-icon">
                  {msg.role === 'user' ? <User size={20} /> : <Bot size={20} />}
                </div>
                <div className="message-content">
                  <div className="message-text">{msg.text}</div>
                  
                  {msg.sources && msg.sources.length > 0 && (
                    <div className="sources-section">
                      <div className="sources-title">
                        <BookOpen size={14} /> Sources referenced:
                      </div>
                      <ul className="sources-list">
                        {msg.sources.map((source, sIdx) => (
                          <li key={sIdx}>{source}</li>
                        ))}
                      </ul>
                    </div>
                  )}
                </div>
              </div>
            );
          })}

          {isLoading && (
            <div className="message-wrapper assistant loading-wrapper">
              <div className="message-icon">
                <Bot size={20} />
              </div>
              <div className="message-content">
                <div className="loading-state">
                  <Loader2 className="spinner" size={24} />
                  <span>Consulting medical knowledge base... (this may take 10-20 seconds)</span>
                </div>
              </div>
            </div>
          )}

          {error && history.length === 0 && (
             <div className="global-error">
                <AlertCircle size={20} />
                {error}
             </div>
          )}
          
          <div ref={messagesEndRef} />
        </div>

        <div className="input-area">
          <form onSubmit={handleSubmit} className="input-form">
            <textarea
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Ask a medical question..."
              disabled={isLoading}
              rows={1}
            />
            <button 
              type="submit" 
              disabled={isLoading || !question.trim()}
              className="send-button"
            >
              {isLoading ? <Loader2 className="spinner" size={20} /> : <Send size={20} />}
            </button>
          </form>
        </div>
      </main>
    </div>
  );
}

export default App;
