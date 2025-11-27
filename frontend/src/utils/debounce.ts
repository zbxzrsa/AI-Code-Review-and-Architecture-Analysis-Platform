// Frontend debouncing for AI requests
export function debounce<T extends (...args: any[]) => any>(
  func: T,
  wait: number
): (...args: Parameters<T>) => void {
  let timeout: NodeJS.Timeout;

  return (...args: Parameters<T>) => {
    clearTimeout(timeout);
    timeout = setTimeout(() => func(...args), wait);
  };
}

// Usage example for AI review
export const debouncedReview = debounce(
  (text: string, channel: string) => {
    // Trigger AI review
    console.log('Reviewing:', text.substring(0, 50) + '...', channel);
  },
  400 // 400ms debounce
);

// Request batching for identical requests
class RequestBatcher {
  private pending = new Map<string, Promise<any>>();
  private batchTimeout = 100; // 100ms batching window

  async batch<T>(key: string, fn: () => Promise<T>): Promise<T> {
    if (this.pending.has(key)) {
      return this.pending.get(key) as Promise<T>;
    }

    const promise = fn();
    this.pending.set(key, promise);

    // Clear from pending after completion
    promise.finally(() => {
      setTimeout(() => this.pending.delete(key), this.batchTimeout);
    });

    return promise;
  }
}

export const requestBatcher = new RequestBatcher();

// Create cache key for requests
export function createCacheKey(text: string, channel: string): string {
  const normalized = text.toLowerCase().trim().replace(/\s+/g, ' ');
  return `${channel}:${Buffer.from(normalized).toString('base64')}`;
}
