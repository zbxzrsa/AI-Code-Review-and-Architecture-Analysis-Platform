import React from 'react';
import { Typography, Space, Avatar } from 'antd';
import { CodeOutlined } from '@ant-design/icons';
import { useTheme } from '../../app/providers/SimpleThemeProvider';

const { Title, Text } = Typography;

// 品牌Logo组件
export const BrandLogo: React.FC<{
  size?: 'small' | 'default' | 'large';
  showText?: boolean;
}> = ({ size = 'default', showText = true }) => {
  const { isDarkMode } = useTheme();

  // 根据尺寸设置大小
  const sizeMap = {
    small: { logoSize: 24, fontSize: 14 },
    default: { logoSize: 32, fontSize: 18 },
    large: { logoSize: 48, fontSize: 24 },
  };

  const { logoSize, fontSize } = sizeMap[size];

  return (
    <Space className="brand-logo" size={8}>
      <Avatar
        size={logoSize}
        icon={<CodeOutlined />}
        style={{
          backgroundColor: isDarkMode ? '#1677ff' : '#1677ff',
          color: '#fff',
        }}
      />
      {showText && (
        <Title
          level={5}
          style={{
            margin: 0,
            fontSize,
            color: isDarkMode ? '#fff' : '#262626',
          }}
        >
          Intelligent Code Review Platform
        </Title>
      )}
    </Space>
  );
};

// 品牌标语组件
export const BrandSlogan: React.FC = () => {
  return (
    <Text className="brand-slogan" style={{ fontSize: 16, color: '#8c8c8c' }}>
      智能化代码审查，高效架构分析
    </Text>
  );
};

// 品牌页脚组件
export const BrandFooter: React.FC = () => {
  const currentYear = new Date().getFullYear();

  return (
    <div className="brand-footer" style={{ textAlign: 'center', padding: '16px 0' }}>
      <Space direction="vertical" size={4}>
        <BrandLogo size="small" />
        <Text type="secondary">© {currentYear} 智能代码审查与架构分析平台. 保留所有权利.</Text>
      </Space>
    </div>
  );
};

// 品牌水印组件
export const BrandWatermark: React.FC<{
  children: React.ReactNode;
}> = ({ children }) => {
  const { isDarkMode } = useTheme();

  return (
    <div style={{ position: 'relative' }}>
      <div
        style={{
          position: 'absolute',
          top: 0,
          left: 0,
          width: '100%',
          height: '100%',
          pointerEvents: 'none',
          zIndex: 0,
          backgroundImage: `url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='200' height='200' viewBox='0 0 200 200'%3E%3Ctext x='50%25' y='50%25' font-family='Arial' font-size='20' fill='${isDarkMode ? 'rgba(255,255,255,0.03)' : 'rgba(0,0,0,0.03)'}' text-anchor='middle' dominant-baseline='middle' transform='rotate(-45, 100, 100)'%3E智能代码审查%3C/text%3E%3C/svg%3E")`,
          backgroundRepeat: 'repeat',
        }}
      />
      {children}
    </div>
  );
};

// 导出所有品牌组件
export default {
  BrandLogo,
  BrandSlogan,
  BrandFooter,
  BrandWatermark,
};
