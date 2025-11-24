import React from 'react';
import { Layout, Button, Space, Dropdown, MenuProps, Switch } from 'antd';
import { UserOutlined, LogoutOutlined, LoginOutlined, BulbOutlined, MoonOutlined } from '@ant-design/icons';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../contexts/AuthContext';
import { BrandLogo } from './ui/BrandElements';
import { useTheme } from '../app/providers/SimpleThemeProvider';

const { Header } = Layout;

const AppHeader: React.FC = () => {
  const navigate = useNavigate();
  const t = (k: string, fb?: string) => fb || k;
  const { isAuthenticated, logout } = useAuth();
  const { mode, toggleTheme } = useTheme();

  const handleLogin = () => {
    navigate('/login');
  };

  const handleLogout = () => {
    logout();
    navigate('/login');
  };

  const items: MenuProps['items'] = [
    {
      key: 'logout',
      label: 'Logout',
      icon: <LogoutOutlined />,
      onClick: handleLogout,
    },
  ];

  return (
    <Header role="banner" aria-label="App Header" style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', backdropFilter: 'saturate(1.2) blur(6px)', background: 'linear-gradient(90deg, rgba(0,0,0,0.85), rgba(24,24,24,0.85))' }}>
      <div className="brand-container">
        <BrandLogo size="default" showText={true} />
      </div>
      <Space align="center">
        <Switch
          checkedChildren={<MoonOutlined />}
          unCheckedChildren={<BulbOutlined />}
          checked={mode === 'dark'}
          onChange={() => toggleTheme()}
          aria-label="Toggle color theme"
          title={mode === 'dark' ? 'Dark' : 'Light'}
        />
        {!isAuthenticated ? (
          <Button className="touch-target" aria-label={'Login'} type="primary" icon={<LoginOutlined />} onClick={handleLogin}>
            {'Login'}
          </Button>
        ) : (
          <Dropdown menu={{ items }} placement="bottomRight">
            <Button className="touch-target" aria-label={'Profile'} type="primary" icon={<UserOutlined />}>{'Profile'}</Button>
          </Dropdown>
        )}
      </Space>
    </Header>
  );
};

export default AppHeader;
