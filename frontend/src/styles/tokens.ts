/**
 * 全局设计令牌与主题配置
 *
 * 包含：
 * - 主题颜色映射
 * - 语义色映射
 * - 间距与尺寸
 * - 字体配置
 * - 阴影定义
 */

// ============ 颜色令牌 ============
export const colorTokens = {
  primary: '#2E90FA',        // 主色 (蓝)
  success: '#12B76A',        // 成功 (绿)
  warning: '#F79009',        // 警告 (橙)
  error: '#F04438',          // 错误 (红)
  info: '#2E90FA',           // 信息 (蓝)

  // 严重级别映射
  critical: '#D92D20',       // 致命
  high: '#F04438',           // 高
  medium: '#F79009',         // 中
  low: '#12B76A',            // 低
  info_level: '#2E90FA',     // 信息
};

// ============ 严重级别颜色映射 ============
export const severityColorMap = {
  CRITICAL: { color: '#D92D20', light: '#FECDD1', label: '致命' },
  HIGH: { color: '#F04438', light: '#FEE2E2', label: '高' },
  MEDIUM: { color: '#F79009', light: '#FFFAEB', label: '中' },
  LOW: { color: '#12B76A', light: '#ECFDF5', label: '低' },
  INFO: { color: '#2E90FA', light: '#EFF8FF', label: '信息' },
};

// ============ 间距与尺寸 ============
export const spacing = {
  xs: 4,     // 4px
  sm: 8,     // 8px
  md: 16,    // 16px
  lg: 24,    // 24px
  xl: 32,    // 32px
  xxl: 48,   // 48px
};

export const borderRadius = {
  xs: 4,     // 4px
  sm: 8,     // 8px - Card 圆角
  md: 12,    // 12px
  lg: 16,    // 16px
};

// ============ 字体配置 ============
export const fontFamily = {
  base: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif',
  mono: '"Monaco", "JetBrains Mono", "Courier New", monospace',
  heading: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif',
};

export const fontSize = {
  xs: 12,
  sm: 13,
  base: 14,
  md: 16,
  lg: 18,
  xl: 20,
  xxl: 24,
  xxxl: 32,
};

export const fontWeight = {
  light: 300,
  normal: 400,
  medium: 500,
  semibold: 600,
  bold: 700,
};

export const lineHeight = {
  tight: 1.2,
  normal: 1.5,
  relaxed: 1.75,
  loose: 2,
};

// ============ 阴影定义 ============
export const shadows = {
  none: 'none',
  xs: '0 1px 2px rgba(16, 24, 48, 0.06), 0 1px 3px rgba(16, 24, 48, 0.1)',
  sm: '0 1px 2px rgba(16, 24, 48, 0.08), 0 1px 3px rgba(16, 24, 48, 0.12), 0 2px 4px rgba(16, 24, 48, 0.08)',
  md: '0 2px 4px rgba(16, 24, 48, 0.08), 0 3px 8px rgba(16, 24, 48, 0.12), 0 4px 16px rgba(16, 24, 48, 0.08)',
  lg: '0 4px 8px rgba(16, 24, 48, 0.1), 0 8px 16px rgba(16, 24, 48, 0.12), 0 16px 32px rgba(16, 24, 48, 0.1)',
  xl: '0 8px 16px rgba(16, 24, 48, 0.12), 0 16px 32px rgba(16, 24, 48, 0.14), 0 32px 64px rgba(16, 24, 48, 0.1)',
};

// ============ 过渡动画 ============
export const transitions = {
  fast: '150ms cubic-bezier(0.4, 0, 0.2, 1)',
  normal: '250ms cubic-bezier(0.4, 0, 0.2, 1)',
  slow: '350ms cubic-bezier(0.4, 0, 0.2, 1)',
};

// ============ 响应式断点 ============
export const breakpoints = {
  xs: 320,
  sm: 640,
  md: 1024,
  lg: 1280,
  xl: 1536,
  xxl: 1920,
};

// ============ Z-index 层级 ============
export const zIndex = {
  hide: -1,
  base: 0,
  dropdown: 1000,
  sticky: 1020,
  fixed: 1030,
  backdrop: 1040,
  offcanvas: 1050,
  modal: 1060,
  popover: 1070,
  tooltip: 1080,
};

// ============ 浅色主题 ============
export const lightTheme = {
  // 背景
  background: {
    base: '#FFFFFF',
    secondary: '#F9FAFB',
    tertiary: '#F3F4F6',
  },
  // 文本
  text: {
    primary: '#101828',
    secondary: '#475569',
    tertiary: '#64748B',
    inverse: '#FFFFFF',
  },
  // 边框
  border: {
    default: '#E5E7EB',
    light: '#F3F4F6',
    dark: '#D1D5DB',
  },
  // 交互
  interactive: {
    hover: 'rgba(46, 144, 250, 0.08)',
    active: 'rgba(46, 144, 250, 0.12)',
    disabled: 'rgba(16, 24, 48, 0.06)',
  },
};

// ============ 深色主题 ============
export const darkTheme = {
  // 背景
  background: {
    base: '#0F172A',
    secondary: '#1E293B',
    tertiary: '#334155',
  },
  // 文本
  text: {
    primary: '#F8FAFC',
    secondary: '#CBD5E1',
    tertiary: '#94A3B8',
    inverse: '#101828',
  },
  // 边框
  border: {
    default: '#334155',
    light: '#475569',
    dark: '#1E293B',
  },
  // 交互
  interactive: {
    hover: 'rgba(46, 144, 250, 0.12)',
    active: 'rgba(46, 144, 250, 0.2)',
    disabled: 'rgba(248, 250, 252, 0.1)',
  },
};

// ============ 组件样式预设 ============
export const componentStyles = {
  card: {
    borderRadius: borderRadius.sm,      // 8px 圆角
    boxShadow: shadows.xs,              // 弱化阴影
    padding: spacing.lg,                // 24px padding
  },
  table: {
    density: 'compact',                 // 紧凑密度
    rowHeight: 40,
  },
  tag: {
    height: 24,
    padding: `${spacing.xs}px ${spacing.sm}px`,
    borderRadius: 4,
    fontSize: fontSize.sm,
  },
  input: {
    height: 40,
    borderRadius: borderRadius.xs,
    padding: `${spacing.sm}px ${spacing.md}px`,
  },
  button: {
    height: 40,
    borderRadius: borderRadius.xs,
    padding: `${spacing.sm}px ${spacing.lg}px`,
    fontSize: fontSize.base,
  },
};

// ============ 工具函数 ============
/**
 * 获取严重级别的颜色
 */
export function getSeverityColor(severity: string): { color: string; light: string } {
  const key = severity.toUpperCase() as keyof typeof severityColorMap;
  return severityColorMap[key] || severityColorMap.INFO;
}

/**
 * 获取严重级别的标签文本
 */
export function getSeverityLabel(severity: string): string {
  const key = severity.toUpperCase() as keyof typeof severityColorMap;
  return severityColorMap[key]?.label || '信息';
}

/**
 * 获取响应式 padding
 */
export function getResponsivePadding(mobile: number, tablet: number, desktop: number) {
  return {
    xs: mobile,
    md: tablet,
    lg: desktop,
  };
}

/**
 * 获取响应式字体大小
 */
export function getResponsiveFontSize(mobile: number, tablet: number, desktop: number) {
  return {
    xs: mobile,
    md: tablet,
    lg: desktop,
  };
}
