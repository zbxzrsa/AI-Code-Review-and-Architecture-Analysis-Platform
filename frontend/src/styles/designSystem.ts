/**
 * 智能代码审查与架构分析平台 - 设计系统
 * 
 * 本文件定义了平台的设计系统，包括颜色、排版、间距、阴影等基础设计元素
 * 以及组件样式变量，确保整个应用的视觉一致性和品牌识别
 */

import { ThemeConfig } from 'antd/es/config-provider/context';

// 品牌色彩系统
export const colors = {
  // 主色调 - 科技蓝
  primary: {
    light: '#4096ff',
    main: '#1677ff',
    dark: '#0958d9',
    contrast: '#ffffff',
  },
  // 辅助色 - 成功
  success: {
    light: '#73d13d',
    main: '#52c41a',
    dark: '#389e0d',
    contrast: '#ffffff',
  },
  // 辅助色 - 警告
  warning: {
    light: '#ffd666',
    main: '#faad14',
    dark: '#d48806',
    contrast: '#ffffff',
  },
  // 辅助色 - 错误
  error: {
    light: '#ff7875',
    main: '#ff4d4f',
    dark: '#d9363e',
    contrast: '#ffffff',
  },
  // 辅助色 - 信息
  info: {
    light: '#91caff',
    main: '#1677ff',
    dark: '#0958d9',
    contrast: '#ffffff',
  },
  // 中性色
  neutral: {
    0: '#ffffff',
    50: '#fafafa',
    100: '#f5f5f5',
    200: '#f0f0f0',
    300: '#d9d9d9',
    400: '#bfbfbf',
    500: '#8c8c8c',
    600: '#595959',
    700: '#434343',
    800: '#262626',
    900: '#1f1f1f',
    1000: '#000000',
  },
  // 功能色
  functional: {
    link: '#1677ff',
    linkHover: '#69b1ff',
    selection: '#e6f4ff',
    disabled: 'rgba(0, 0, 0, 0.25)',
    disabledBg: '#f5f5f5',
    mask: 'rgba(0, 0, 0, 0.45)',
    divider: '#f0f0f0',
    border: '#d9d9d9',
    backgroundLight: '#f5f5f5',
    backgroundDark: '#141414',
  },
  // 数据可视化色板
  dataViz: [
    '#1677ff', '#52c41a', '#faad14', '#eb2f96', '#722ed1',
    '#13c2c2', '#fadb14', '#a0d911', '#fa541c', '#2f54eb',
  ],
};

// 排版系统
export const typography = {
  fontFamily: `'PingFang SC', 'Microsoft YaHei', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 
               'Helvetica Neue', Arial, 'Noto Sans', sans-serif, 'Apple Color Emoji', 
               'Segoe UI Emoji', 'Segoe UI Symbol', 'Noto Color Emoji'`,
  codeFamily: `'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, Courier, monospace`,
  fontSize: {
    xs: 12,
    sm: 14,
    base: 16,
    lg: 18,
    xl: 20,
    xxl: 24,
    xxxl: 30,
    display: 38,
  },
  fontWeight: {
    light: 300,
    regular: 400,
    medium: 500,
    semibold: 600,
    bold: 700,
  },
  lineHeight: {
    tight: 1.2,
    base: 1.5,
    relaxed: 1.75,
  },
  letterSpacing: {
    tighter: '-0.05em',
    tight: '-0.025em',
    normal: '0',
    wide: '0.025em',
    wider: '0.05em',
  },
};

// 间距系统
export const spacing = {
  0: 0,
  1: 4,
  2: 8,
  3: 12,
  4: 16,
  5: 20,
  6: 24,
  7: 28,
  8: 32,
  9: 36,
  10: 40,
  12: 48,
  16: 64,
  20: 80,
  24: 96,
  32: 128,
};

// 圆角系统
export const borderRadius = {
  none: 0,
  xs: 2,
  sm: 4,
  md: 6,
  lg: 8,
  xl: 12,
  xxl: 16,
  full: 9999,
};

// 阴影系统
export const shadows = {
  none: 'none',
  xs: '0 1px 2px rgba(0, 0, 0, 0.05)',
  sm: '0 1px 3px rgba(0, 0, 0, 0.1), 0 1px 2px rgba(0, 0, 0, 0.06)',
  md: '0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06)',
  lg: '0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05)',
  xl: '0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04)',
  xxl: '0 25px 50px -12px rgba(0, 0, 0, 0.25)',
  inner: 'inset 0 2px 4px 0 rgba(0, 0, 0, 0.06)',
};

// 动画系统
export const animations = {
  durations: {
    fastest: 100,
    fast: 200,
    normal: 300,
    slow: 500,
    slowest: 800,
  },
  easings: {
    easeInOut: 'cubic-bezier(0.4, 0, 0.2, 1)',
    easeOut: 'cubic-bezier(0.0, 0, 0.2, 1)',
    easeIn: 'cubic-bezier(0.4, 0, 1, 1)',
    sharp: 'cubic-bezier(0.4, 0, 0.6, 1)',
  },
};

// 断点系统 - 响应式设计
export const breakpoints = {
  xs: 480,
  sm: 576,
  md: 768,
  lg: 992,
  xl: 1200,
  xxl: 1600,
};

// Ant Design主题配置
export const designSystemTheme: ThemeConfig = {
  token: {
    colorPrimary: colors.primary.main,
    colorSuccess: colors.success.main,
    colorWarning: colors.warning.main,
    colorError: colors.error.main,
    colorInfo: colors.info.main,
    colorTextBase: colors.neutral[900],
    colorBgBase: colors.neutral[0],
    fontFamily: typography.fontFamily,
    fontSize: typography.fontSize.base,
    borderRadius: borderRadius.md,
    wireframe: false,
  },
  components: {
    Layout: {
      headerBg: colors.neutral[0],
      bodyBg: colors.neutral[50],
      siderBg: colors.neutral[0],
    },
    Menu: {
      itemSelectedBg: colors.functional.selection,
      itemSelectedColor: colors.primary.main,
      itemHoverColor: colors.primary.main,
      itemHoverBg: colors.functional.selection,
      activeBarWidth: 3,
      itemMarginInline: spacing[3],
    },
    Card: {
      colorBorderSecondary: colors.functional.border,
      boxShadowTertiary: shadows.sm,
      borderRadiusLG: borderRadius.md,
    },
    Button: {
      borderRadius: borderRadius.md,
      controlHeight: 36,
      controlHeightLG: 44,
      controlHeightSM: 28,
      paddingContentHorizontal: spacing[4],
    },
    Input: {
      controlHeight: 36,
      controlHeightLG: 44,
      controlHeightSM: 28,
      borderRadius: borderRadius.md,
    },
    Select: {
      controlHeight: 36,
      controlHeightLG: 44,
      controlHeightSM: 28,
      borderRadius: borderRadius.md,
    },
    Table: {
      borderRadius: borderRadius.md,
      colorBgContainer: colors.neutral[0],
      headerBg: colors.neutral[50],
    },
    Tabs: {
      inkBarColor: colors.primary.main,
      itemSelectedColor: colors.primary.main,
      itemHoverColor: colors.primary.light,
    },
    Modal: {
      borderRadiusLG: borderRadius.lg,
      paddingContentHorizontalLG: spacing[6],
    },
    Drawer: {
      paddingContentHorizontalLG: spacing[6],
    },
    Message: {
      borderRadiusLG: borderRadius.lg,
    },
    Notification: {
      borderRadiusLG: borderRadius.lg,
    },
    Tooltip: {
      colorBgSpotlight: colors.neutral[800],
      borderRadius: borderRadius.sm,
    },
  },
};

// 导出默认设计系统
export default designSystemTheme;