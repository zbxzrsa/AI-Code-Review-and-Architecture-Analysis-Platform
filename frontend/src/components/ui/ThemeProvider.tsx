// 兼容导出：将旧的 components/ui ThemeProvider 重定向到集中式 app/providers SimpleThemeProvider
export { SimpleThemeProvider as ThemeProvider, useTheme } from '../../app/providers/SimpleThemeProvider';
export { SimpleThemeProvider as default } from '../../app/providers/SimpleThemeProvider';
