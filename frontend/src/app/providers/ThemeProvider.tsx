import { ConfigProvider, theme as antdTheme } from 'antd';
import React, { createContext, useContext, useState, useEffect } from 'react';

import {
    colorTokens,
    darkTheme,
    lightTheme,
    spacing,
    borderRadius,
    fontFamily,
    shadows,
} from '../../styles/tokens';

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

export function ThemeProvider({ children }: { children: React.ReactNode }) {
    const [mode, setMode] = useState<ThemeMode>('light');

    useEffect(() => {
        const saved = localStorage.getItem('theme-mode') as ThemeMode | null;
        const systemPreference = window.matchMedia('(prefers-color-scheme: dark)').matches
            ? 'dark'
            : 'light';
        setMode(saved || systemPreference);
    }, []);

    const setThemeMode = (m: ThemeMode) => {
        const resolvedMode =
            m === 'system'
                ? window.matchMedia('(prefers-color-scheme: dark)').matches
                    ? 'dark'
                    : 'light'
                : m;
        setMode(m);
        localStorage.setItem('theme-mode', m);
        document.documentElement.setAttribute('data-theme', resolvedMode);
    };

    const isDarkMode =
        mode === 'dark' ||
        (mode === 'system' &&
            window.matchMedia &&
            window.matchMedia('(prefers-color-scheme: dark)').matches);

    const themeConfig = {
        token: {
            colorPrimary: colorTokens.primary,
            colorSuccess: colorTokens.success,
            colorWarning: colorTokens.warning,
            colorError: colorTokens.error,
            colorInfo: colorTokens.info,
            fontFamily: fontFamily.base,
            fontSize: 14,
            fontWeightStrong: 600,
            lineHeight: 1.5,
            lineHeightHeading1: 1.2,
            lineHeightHeading2: 1.35,

            borderRadius: borderRadius.md,
            borderRadiusLG: borderRadius.lg,
            borderRadiusSM: borderRadius.sm,
            borderRadiusXS: 2,

            margin: spacing.md,
            marginXS: spacing.xs,
            marginSM: spacing.sm,
            marginLG: spacing.lg,
            marginXL: spacing.xl,

            padding: spacing.md,
            paddingXS: spacing.xs,
            paddingSM: spacing.sm,
            paddingLG: spacing.lg,
            paddingXL: spacing.xl,

            controlHeight: 40,
            controlHeightLG: 48,
            controlHeightSM: 32,
            controlHeightXS: 24,

            boxShadow: shadows.xs,
            boxShadowSecondary: shadows.none,

            motion: true,
            colorBgContainer: isDarkMode ? darkTheme.background.base : lightTheme.background.base,
            colorBgElevated: isDarkMode
                ? darkTheme.background.secondary
                : lightTheme.background.secondary,
            colorBgLayout: isDarkMode
                ? darkTheme.background.tertiary
                : lightTheme.background.tertiary,
            colorText: isDarkMode ? darkTheme.text.primary : lightTheme.text.primary,
            colorTextSecondary: isDarkMode ? darkTheme.text.secondary : lightTheme.text.secondary,
            colorBorder: isDarkMode ? darkTheme.border.default : lightTheme.border.default,
        },
        algorithm: isDarkMode ? antdTheme.darkAlgorithm : antdTheme.defaultAlgorithm,
        components: {
            Card: {
                controlHeight: 40,
                borderRadiusLG: borderRadius.sm,
                boxShadowSecondary: shadows.none,
                boxShadow: shadows.xs,
            },
            Table: {
                controlHeight: 36,
            },
            Tag: {
                borderRadiusSM: 4,
                controlHeight: 24,
                colorBgContainer: isDarkMode
                    ? darkTheme.background.secondary
                    : lightTheme.background.secondary,
                colorPrimary: colorTokens.primary,
            },
        },
    };

    const contextValue: ThemeContextType = {
        mode,
        toggleTheme: () => setThemeMode(mode === 'dark' ? 'light' : 'dark'),
        setTheme: setThemeMode,
        themeMode: mode,
        setThemeMode,
        isDarkMode,
    };

    return (
        <ThemeContext.Provider value={contextValue}>
            <ConfigProvider theme={themeConfig}>{children}</ConfigProvider>
        </ThemeContext.Provider>
    );
}

export function useTheme() {
    const ctx = useContext(ThemeContext);
    if (!ctx) throw new Error('useTheme must be used within ThemeProvider');
    return ctx;
}
