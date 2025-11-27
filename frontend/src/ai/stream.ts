// EventSource-based streaming client for AI reviews
export class AIStreamClient {
  private eventSource: EventSource | null = null;
  private onChunk?: (chunk: string) => void;
  private onError?: (error: string) => void;
  private onComplete?: () => void;

  constructor(
    private url: string,
    private headers: Record<string, string> = {}
  ) {}

  on(event: 'chunk' | 'error' | 'complete', callback: Function) {
    switch (event) {
      case 'chunk':
        this.onChunk = callback as (chunk: string) => void;
        break;
      case 'error':
        this.onError = callback as (error: string) => void;
        break;
      case 'complete':
        this.onComplete = callback as () => void;
        break;
    }
  }

  async start(text: string, channel: string = 'stable'): Promise<void> {
    try {
      // Use fetch with streaming for better compatibility
      const response = await fetch(this.url, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-AI-Channel': channel,
          ...this.headers,
        },
        body: JSON.stringify({ text }),
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      const reader = response.body?.getReader();
      const decoder = new TextDecoder();

      if (!reader) {
        throw new Error('Response body is not readable');
      }

      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();

        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            try {
              const data = JSON.parse(line.slice(6));

              if (data.chunk) {
                this.onChunk?.(data.chunk);
              } else if (data.error) {
                this.onError?.(data.error);
                return;
              } else if (data.done) {
                this.onComplete?.();
                return;
              }
            } catch (e) {
              console.warn('Failed to parse SSE data:', line);
            }
          } else if (line.startsWith(': ')) {
            // Heartbeat - ignore
            continue;
          }
        }
      }
    } catch (error) {
      this.onError?.(error instanceof Error ? error.message : 'Unknown error');
    }
  }

  stop() {
    if (this.eventSource) {
      this.eventSource.close();
      this.eventSource = null;
    }
  }
}

// React hook for streaming AI reviews
export function useAIStream() {
  const [isStreaming, setIsStreaming] = useState(false);
  const [response, setResponse] = useState('');
  const [error, setError] = useState<string | null>(null);

  const streamReview = useCallback(async (text: string, channel: string = 'stable') => {
    setIsStreaming(true);
    setResponse('');
    setError(null);

    const client = new AIStreamClient('/ai/review/stream');

    client.on('chunk', (chunk: string) => {
      setResponse(prev => prev + chunk);
    });

    client.on('error', (error: string) => {
      setError(error);
      setIsStreaming(false);
    });

    client.on('complete', () => {
      setIsStreaming(false);
    });

    await client.start(text, channel);
  }, []);

  return {
    isStreaming,
    response,
    error,
    streamReview,
  };
}

// Fallback non-streaming function
export async function review(text: string, channel: 'stable' | 'next' | 'legacy' = 'stable') {
  if (channel === 'next') {
    // Try WebLLM first, fallback to server
    try {
      return await reviewInBrowser(text);
    } catch (e) {
      console.warn('WebLLM failed, falling back to server:', e);
    }
  }

  const res = await fetch('/ai/review', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'X-AI-Channel': channel,
    },
    body: JSON.stringify({ text }),
  });

  if (!res.ok) {
    throw new Error(`HTTP ${res.status}: ${res.statusText}`);
  }

  const data = await res.json();
  return data.result;
}
