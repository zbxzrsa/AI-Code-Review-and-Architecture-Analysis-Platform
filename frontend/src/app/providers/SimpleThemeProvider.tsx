import { ConfigProvider, theme as antdTheme } from 'antd';
import React, { createContext, useContext } from 'react';

interface ThemeProviderProps {
  children: React.ReactNode;
}

export type ThemeMode = 'light' | 'dark' | 'system';

interface ThemeContextType {
    mode: ThemeMode;
    toggleTheme: () => void;
    setTheme: (mode: ThemeMode) => void;
    themeMode?: ThemeMode;
    setThemeMode?: (mode: ThemeMode) => void;
    isDarkMode?: boolean;
}

const ThemeContext = createContext<ThemeContextType | undefined>(undefined);

export function SimpleThemeProvider({ children }: ThemeProviderProps) {
  const contextValue: ThemeContextType = {
    mode: 'light',
    toggleTheme: () => {
      // Simple toggle implementation
      document.documentElement.setAttribute('data-theme', 'light');
    },
    setTheme: (mode: ThemeMode) => {
      document.documentElement.setAttribute('data-theme', mode);
    },
    isDarkMode: false,
  };

  return (
    <ThemeContext.Provider value={contextValue}>
      <ConfigProvider
        theme={{
          algorithm: antdTheme.defaultAlgorithm,
          token: {
            colorPrimary: '#1677ff',
            borderRadius: 6,
            fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, "Noto Sans", sans-serif, "Apple Color Emoji", "Segoe UI Emoji", "Segoe UI Symbol", "Noto Color Emoji"',
          },
        }}
      >
        {children}
      </ConfigProvider>
    </ThemeContext.Provider>
  );
}

export function useTheme() {
  const context = useContext(ThemeContext);
  if (context === undefined) {
    throw new Error('useTheme must be used within a ThemeProvider');
  }
  return context;
}
