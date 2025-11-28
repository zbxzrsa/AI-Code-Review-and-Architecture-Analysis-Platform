import { useState, useEffect } from 'react';

interface CacheConfig {
  maxAge: number;
  maxSize: number;
}

interface CacheItem {
  data: any;
  timestamp: number;
  expiresAt: number;
}

interface UseLocalStorageReturn {
  value: T | null;
  setItem: (key: string, value: T) => void;
  removeItem: (key: string) => void;
  clear: () => void;
  isAvailable: boolean;
  size: number;
}

const CACHE_CONFIG: CacheConfig = {
  maxAge: 30 * 60 * 1000, // 30 minutes
  maxSize: 50 * 1024 * 1024, // 50MB
};

const useLocalStorage = <T>(key: string, defaultValue: T): UseLocalStorageReturn => {
  // Check if localStorage is available
  const isAvailable = typeof window !== 'undefined' && typeof window.localStorage !== 'undefined';
  
  // Get initial value from localStorage or use default
  const getInitialValue = (): T => {
    if (!isAvailable) return defaultValue;
    try {
      const item = window.localStorage.getItem(key);
      return item ? JSON.parse(item) : defaultValue;
    } catch {
      return defaultValue;
    }
  };

  const [value, setValue] = useState<T>(getInitialValue);

  // Set value in localStorage
  const setItem = (key: string, value: T): void => {
    if (!isAvailable) return;
    try {
      const serializedValue = JSON.stringify(value);
      window.localStorage.setItem(key, serializedValue);
    } catch (error) {
      console.error('Failed to save to localStorage:', error);
    }
  };

  // Remove item from localStorage
  const removeItem = (key: string): void => {
    if (!isAvailable) return;
    try {
      window.localStorage.removeItem(key);
    } catch (error) {
      console.error('Failed to remove from localStorage:', error);
    }
  };

  // Clear all cache
  const clear = (): void => {
    if (!isAvailable) return;
    try {
      const keys = Object.keys(window.localStorage);
      keys.forEach(key => window.localStorage.removeItem(key));
    } catch (error) {
      console.error('Failed to clear localStorage:', error);
    }
  };

  // Get cache size
  const getCacheSize = (): number => {
    if (!isAvailable) return 0;
    try {
      let size = 0;
      for (let i = 0; i < window.localStorage.length; i++) {
        const key = window.localStorage.key(i);
        const item = window.localStorage.getItem(key);
        if (item) {
          size += new Blob([item]).size;
        }
      }
      return size;
    } catch (error) {
      return 0;
    }
  };

  // Clean expired items
  const cleanExpiredItems = (): void => {
    if (!isAvailable) return;
    try {
      const now = Date.now();
      const keys = Object.keys(window.localStorage);
      
      keys.forEach(key => {
        const item = window.localStorage.getItem(key);
        if (item) {
          const parsedItem = JSON.parse(item);
          if (parsedItem.timestamp && (now - parsedItem.timestamp > CACHE_CONFIG.maxAge)) {
            window.localStorage.removeItem(key);
          }
        }
      }
    } catch (error) {
      console.error('Failed to clean expired items:', error);
    }
  };

  // Auto-cleanup expired items on mount
  useEffect(() => {
    const interval = setInterval(cleanExpiredItems, CACHE_CONFIG.maxAge);
    return () => clearInterval(interval);
  }, [isAvailable]);

  return {
    value,
    setValue,
    removeItem,
    clear,
    isAvailable,
    size: getCacheSize(),
    cleanExpiredItems,
  };
};

export default useLocalStorage;