import React from 'react';
import { Button, Tooltip, Dropdown } from 'antd';
import { BulbOutlined, BulbFilled, SettingOutlined } from '@ant-design/icons';
import { useTheme } from '../../app/providers/SimpleThemeProvider';

// 主题切换器组件
const ThemeSwitcher: React.FC<{ mode?: 'icon' | 'dropdown' }> = ({ mode = 'dropdown' }) => {
  const { themeMode, setThemeMode, isDarkMode } = useTheme();

  if (!setThemeMode) return null;

  // 图标模式的主题切换器
  if (mode === 'icon') {
    return (
      <Tooltip title={isDarkMode ? '切换到亮色模式' : '切换到暗色模式'}>
        <Button
          type="text"
          icon={isDarkMode ? <BulbOutlined /> : <BulbFilled />}
          onClick={() => setThemeMode?.(isDarkMode ? 'light' : 'dark')}
          aria-label="切换主题"
          className="theme-switcher-btn"
        />
      </Tooltip>
    );
  }

  // 下拉菜单模式的主题切换器
  return (
    <Dropdown
      menu={{
        items: [
          {
            key: 'light',
            label: '亮色模式',
            icon: <BulbOutlined />,
            onClick: () => setThemeMode?.('light'),
          },
          {
            key: 'dark',
            label: '暗色模式',
            icon: <BulbFilled />,
            onClick: () => setThemeMode?.('dark'),
          },
          {
            key: 'system',
            label: '跟随系统',
            icon: <SettingOutlined />,
            onClick: () => setThemeMode?.('system'),
          },
        ],
        selectedKeys: themeMode ? [themeMode] : [],
      }}
      placement="bottomRight"
      trigger={['click']}
    >
      <Button
        type="text"
        icon={isDarkMode ? <BulbOutlined /> : <BulbFilled />}
        aria-label="主题设置"
        className="theme-switcher-btn"
      />
    </Dropdown>
  );
};

export default ThemeSwitcher;
