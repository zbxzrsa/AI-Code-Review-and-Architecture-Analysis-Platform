import type { ThemeConfig } from 'antd/es/config-provider/context';

// 统一视觉设计语言的主题代币配置
export const appTheme: ThemeConfig = {
  token: {
    colorPrimary: '#1677ff',
    colorSuccess: '#52c41a',
    colorWarning: '#faad14',
    colorError: '#f5222d',
    colorInfo: '#1677ff',
    borderRadius: 8,
    fontSize: 14,
    wireframe: false,
    fontFamily: "Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, 'Noto Sans'",
    fontWeightStrong: 600,
    lineHeight: 1.5,
    motionDurationFast: '0.2s',
    motionDurationMid: '0.3s',
    motionEaseInOut: 'cubic-bezier(0.4, 0, 0.2, 1)',
  },
  components: {
    Layout: {
      headerBg: '#001529',
    },
    Button: {
      borderRadius: 6,
    },
    Card: {
      borderRadiusLG: 10,
    },
    Menu: {
      itemBorderRadius: 6,
    },
    Tabs: {
      cardBg: '#ffffff',
    },
  },
};

export default appTheme;