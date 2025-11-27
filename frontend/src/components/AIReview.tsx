import React, { useState, useCallback } from 'react';
import { useAIStream } from '../ai/stream';

interface AIReviewProps {
  initialText?: string;
  onReviewComplete?: (review: string) => void;
}

export function AIReview({ initialText = '', onReviewComplete }: AIReviewProps) {
  const [text, setText] = useState(initialText);
  const [channel, setChannel] = useState<'stable' | 'next' | 'legacy'>('stable');
  const { isStreaming, response, error, streamReview } = useAIStream();

  const handleSubmit = useCallback(async () => {
    if (!text.trim()) return;
    await streamReview(text, channel);
  }, [text, channel, streamReview]);

  React.useEffect(() => {
    if (response && !isStreaming) {
      onReviewComplete?.(response);
    }
  }, [response, isStreaming, onReviewComplete]);

  return (
    <div className="ai-review">
      <div className="controls">
        <select
          value={channel}
          onChange={e => setChannel(e.target.value as any)}
          disabled={isStreaming}
        >
          <option value="stable">Stable (Safe)</option>
          <option value="next">Next (Experimental)</option>
          <option value="legacy">Legacy (Conservative)</option>
        </select>

        <button
          onClick={handleSubmit}
          disabled={isStreaming || !text.trim()}
          className="review-button"
        >
          {isStreaming ? 'Reviewing...' : 'Review Code'}
        </button>
      </div>

      <div className="input-section">
        <textarea
          value={text}
          onChange={e => setText(e.target.value)}
          placeholder="Paste your code here for review..."
          disabled={isStreaming}
          rows={10}
          className="code-input"
        />
      </div>

      {error && (
        <div className="error">
          <strong>Error:</strong> {error}
        </div>
      )}

      {response && (
        <div className="response-section">
          <h3>AI Review ({channel})</h3>
          <div className="response-content">
            {response.split('\n').map((line, i) => (
              <div key={i} className="response-line">
                {line || <br />}
              </div>
            ))}
          </div>
          {isStreaming && <div className="streaming-indicator">▋</div>}
        </div>
      )}
    </div>
  );
}
