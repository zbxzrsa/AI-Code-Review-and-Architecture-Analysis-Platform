// Enhanced WebLLM client with capability detection and fallback
interface WebLLMConfig {
  model: string;
  temperature: number;
  maxTokens: number;
}

interface WebLLMCapabilities {
  webGPU: boolean;
  sharedArrayBuffer: boolean;
  webAssembly: boolean;
  indexedDB: boolean;
}

class WebLLMClient {
  private capabilities: WebLLMCapabilities;
  private engine: any = null;
  private isInitialized = false;
  private modelCache: Map<string, any> = new Map();

  constructor() {
    this.capabilities = this.detectCapabilities();
  }

  private detectCapabilities(): WebLLMCapabilities {
    return {
      webGPU: !!(navigator as any).gpu,
      sharedArrayBuffer: typeof SharedArrayBuffer !== 'undefined',
      webAssembly: typeof WebAssembly !== 'undefined',
      indexedDB: typeof indexedDB !== 'undefined',
    };
  }

  get isSupported(): boolean {
    return (
      this.capabilities.webGPU &&
      this.capabilities.sharedArrayBuffer &&
      this.capabilities.webAssembly &&
      this.capabilities.indexedDB
    );
  }

  async initialize(config: WebLLMConfig): Promise<void> {
    if (!this.isSupported) {
      throw new Error('WebLLM not supported: missing required capabilities');
    }

    if (this.isInitialized) {
      return;
    }

    try {
      // Load WebLLM dynamically
      const { CreateMLCEngine } = await import('@mlc-ai/web-llm');

      // Check cache first
      const cacheKey = `${config.model}_${config.temperature}`;
      if (this.modelCache.has(cacheKey)) {
        this.engine = this.modelCache.get(cacheKey);
        this.isInitialized = true;
        return;
      }

      // Initialize engine
      this.engine = await CreateMLCEngine(config.model, {
        temperature: config.temperature,
        maxTokens: config.maxTokens,
      });

      // Cache the engine
      this.modelCache.set(cacheKey, this.engine);
      this.isInitialized = true;

      // Persist to IndexedDB for future sessions
      this.saveToCache(cacheKey, config);
    } catch (error) {
      throw new Error(`WebLLM initialization failed: ${error}`);
    }
  }

  async generate(prompt: string): Promise<string> {
    if (!this.isInitialized) {
      throw new Error('WebLLM not initialized');
    }

    try {
      const response = await this.engine.chat.completions.create({
        messages: [
          {
            role: 'system',
            content: 'You are experimental AI reviewer. Try novel suggestions, but annotate risks.',
          },
          {
            role: 'user',
            content: `Review this code change and suggest improvements:\n${prompt}`,
          },
        ],
        temperature: 0.6,
        max_tokens: 800,
      });

      return response.choices[0]?.message?.content || '';
    } catch (error) {
      throw new Error(`WebLLM generation failed: ${error}`);
    }
  }

  private async saveToCache(key: string, config: WebLLMConfig): Promise<void> {
    if (!this.capabilities.indexedDB) return;

    try {
      const request = indexedDB.open('WebLLMCache', 1);

      request.onupgradeneeded = event => {
        const db = (event.target as IDBOpenDBRequest).result;
        if (!db.objectStoreNames.contains('models')) {
          db.createObjectStore('models');
        }
      };

      request.onsuccess = event => {
        const db = (event.target as IDBOpenDBRequest).result;
        const transaction = db.transaction(['models'], 'readwrite');
        const store = transaction.objectStore('models');
        store.put({ key, config, timestamp: Date.now() });
      };
    } catch (error) {
      console.warn('Failed to cache WebLLM model:', error);
    }
  }

  getCapabilities(): WebLLMCapabilities {
    return { ...this.capabilities };
  }
}

// Singleton instance
let webLLMClient: WebLLMClient | null = null;

export async function getWebLLMClient(): Promise<WebLLMClient> {
  if (!webLLMClient) {
    webLLMClient = new WebLLMClient();
  }
  return webLLMClient;
}

// Enhanced review function with fallback
export async function reviewInBrowser(text: string): Promise<string> {
  try {
    const client = await getWebLLMClient();

    if (!client.isSupported) {
      throw new Error('WebLLM not supported in this browser');
    }

    await client.initialize({
      model: 'qwen2-1.5b-instruct-q4f16_1-MLC',
      temperature: 0.6,
      maxTokens: 800,
    });

    return await client.generate(text);
  } catch (error) {
    console.warn('WebLLM failed, falling back to server:', error);

    // Fallback to server-side next channel
    const response = await fetch('/ai/review', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'X-AI-Channel': 'next',
        'X-WebLLM-Fallback': 'true',
      },
      body: JSON.stringify({ text }),
    });

    if (!response.ok) {
      throw new Error(`Fallback request failed: ${response.statusText}`);
    }

    const data = await response.json();
    return data.result;
  }
}

// React hook for WebLLM capability detection
export function useWebLLMCapabilities() {
  const [capabilities, setCapabilities] = React.useState<WebLLMCapabilities | null>(null);
  const [isSupported, setIsSupported] = React.useState<boolean | null>(null);

  React.useEffect(() => {
    const detect = async () => {
      try {
        const client = await getWebLLMClient();
        const caps = client.getCapabilities();
        setCapabilities(caps);
        setIsSupported(client.isSupported);
      } catch (error) {
        setCapabilities({
          webGPU: false,
          sharedArrayBuffer: false,
          webAssembly: false,
          indexedDB: false,
        });
        setIsSupported(false);
      }
    };

    detect();
  }, []);

  return { capabilities, isSupported };
}
