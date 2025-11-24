/**
 * 应用壳与导航系统
 * 包含：
 * - AppShell: 主容器 (Layout + Sider + Header + Content + Footer)
 * - PageHeader: 页面标题区
 * - RightPanel: 可折叠的右侧 AI 天面板
 * - NotificationsPanel: 通知面板
 * - UserMenu: 用户菜单
 * - QuickActions: 快速操作
 */
import React, { useState, useEffect } from 'react';
import { Layout, Menu, Button, Space, Dropdown, Avatar, Badge, Drawer, Input, Tooltip, Popover } from 'antd';
import { useNavigate } from 'react-router-dom';
import NotificationsPanel from '../notifications/NotificationsPanel';

// 图标组件
const LogoWrapper = ({ children, onClick, className = "" }) => (
  <div 
    className={`logo-icon ${className}`}
    onClick={onClick}
    style={{ cursor: 'pointer' }}
  >
    CI
    {!children && <span>CodeInsight</span>}
  </div>
);

// 主菜单项
const MAIN_MENU_ITEMS = [
  {
    key: 'dashboard',
    label: 'Dashboard',
    icon: 'DashboardOutlined',
    path: '/dashboard',
  },
  {
    key: 'projects',
    label: 'Projects',
    icon: 'FolderOutlined',
    path: '/projects',
  },
  {
    key: 'sessions',
    label: 'Sessions',
    icon: 'ClockCircleOutlined',
    path: '/sessions',
  },
  {
    key: 'versions',
    label: 'Versions',
    icon: 'FileTextOutlined',
    path: '/versions',
  },
  {
    key: 'search',
    label: 'Search',
    icon: 'SearchOutlined',
    path: '/search',
  },
  {
    key: 'baselines',
    label: 'Baselines',
    icon: 'GitlabOutlined',
    path: '/baselines',
  },
  {
    key: 'monitoring',
    label: 'Monitoring',
    icon: 'MonitorOutlined',
    path: '/monitoring',
  },
];

// 快速操作项
const QUICK_ACTIONS = [
  {
    key: 'new-project',
    label: 'New Project',
    icon: 'PlusOutlined',
    action: 'new-project',
  },
  {
    key: 'import-project',
    label: 'Import Project',
    icon: 'ImportOutlined',
    action: 'import-project',
  },
  {
    key: 'run-analysis',
    label: 'Run Analysis',
    icon: 'PlayCircleOutlined',
    action: 'run-analysis',
  },
];

// 用户菜单项
const userMenuItems = [
  {
    key: 'profile',
    label: 'Profile',
    icon: 'UserOutlined',
  },
  {
    key: 'github',
    label: 'GitHub Connect',
    icon: 'GithubOutlined',
  },
  {
    key: 'settings',
    label: 'Settings',
    icon: 'SettingOutlined',
  },
  {
    key: 'help',
    label: 'Help',
    icon: 'QuestionCircleOutlined',
  },
  {
    key: 'logout',
    label: 'Logout',
    icon: 'LogoutOutlined',
    danger: true,
  },
];

// ============ AppShell 主组件 ============
const AppShell: React.FC = () => {
  const navigate = useNavigate();
  const { mode, toggleTheme } = useTheme();
  const [collapsed, setCollapsed] = useState(false);
  const [rightPanelOpen, setRightPanelOpen] = useState(false);
  const [notificationsOpen, setNotificationsOpen] = useState(false);
  const [runAnalysisVisible, setRunAnalysisVisible] = useState(false);
  const [selectedMenuItem, setSelectedMenuItem] = useState('dashboard');
  const [notifications, setNotifications] = useState(3); // 模拟通知数

  // 初始化右侧面板状态（持久化）
  useEffect(() => {
    try {
      const raw = localStorage.getItem('app.rightPanelOpen');
      if (raw !== null) {
        setRightPanelOpen(raw === 'true');
      }
    } catch (e) {
      // ignore
    }
  }, []);

  // 处理菜单项点击
  const handleMenuClick = (key: string) => {
    const menuItem = MAIN_MENU_ITEMS.find((item) => item.key === key);
    if (menuItem) {
      setSelectedMenuItem(key);
      navigate(menuItem.path);
    }
  };

  // 处理快速操作
  const handleQuickAction = (action: string) => {
    switch (action) {
      case 'new-project':
        navigate('/projects/new');
        break;
      case 'import-project':
        navigate('/projects/import');
        break;
      case 'run-analysis':
        setRunAnalysisVisible(true);
        break;
      default:
        break;
    }
  };

  return (
    <>
      <LayoutWrapper>
      {/* 侧边栏 */}
      <Layout.Sider
        collapsed={collapsed}
        onCollapse={(collapsed) => setCollapsed(!collapsed)}
        collapsible
        trigger={null}
        width={240}
        style={{
          position: 'fixed',
          left: 0,
          top: 0,
          bottom: 0,
          zIndex: 100,
          height: '100vh',
          overflow: 'auto',
          borderRight: '1px solid var(--border-default)',
        }}
      >
        {/* Logo区域 */}
        <div style={{ padding: '16px', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
          <LogoWrapper onClick={() => navigate('/dashboard')}>
            <div className="logo-icon">CI</div>
            {!collapsed && <span>CodeInsight</span>}
          </LogoWrapper>
        </div>

        {/* 菜单 */}
        <Menu
          mode="inline"
          selectedKeys={[selectedMenuItem]}
          onClick={(info) => handleMenuClick(info.key)}
          items={MAIN_MENU_ITEMS}
          style={{ border: 'none' }}
        />

        {/* 快速操作 */}
        <div style={{ marginTop: '16px' }}>
          <QuickActionsWrapper size="small">
            {QUICK_ACTIONS.map((action) => (
              <Tooltip key={action.key} title={action.label}>
                <Button
                  type="default"
                  icon={action.icon}
                  onClick={() => handleQuickAction(action.action)}
                >
                  {action.label}
                </Button>
              </Tooltip>
            ))}
          </QuickActionsWrapper>
      </div>
      </Layout.Sider>

      {/* 顶部导航栏 */}
      <Layout.Header>
        <HeaderLeftWrapper>
          {/* 菜单折叠按钮 */}
          <Button
            type="text"
            icon={collapsed ? <MenuUnfoldOutlined /> : <MenuFoldOutlined />}
            onClick={() => setCollapsed(!collapsed)}
          />

          {/* Logo和标题 */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <LogoWrapper onClick={() => navigate('/dashboard')}>
              <div className="logo-icon">CI</div>
              {!collapsed && <span>CodeInsight</span>}
            </LogoWrapper>
            <span style={{ fontSize: collapsed ? '14px' : '16px', fontWeight: 'bold' }}>
              {!collapsed && 'CodeInsight'}
            </span>
          </div>
        </HeaderLeftWrapper>

        {/* 右侧操作区 */}
        <HeaderRightWrapper>
          {/* 主题切换 */}
          <Tooltip title={mode === 'light' ? 'Switch to dark theme' : 'Switch to light theme'}>
            <Button type="text" onClick={toggleTheme}>
              {mode === 'light' ? '🌙️' : '🌚️'}
            </Button>
          </Tooltip>

          {/* 通知按钮 */}
          <Tooltip title="Notifications">
            <Badge count={notifications} offset={[-5, 5]} size="small">
              <BellOutlined />
            </Badge>
            <Button
              type="text"
              icon={<BellOutlined />}
              onClick={() => setNotificationsOpen(true)}
            />
          </Tooltip>

          {/* 用户头像 */}
          <Dropdown
            menu={{
              items: userMenuItems,
              selectedKeys: [selectedMenuItem],
              onClick: (info) => {
                setSelectedMenuItem(info.key);
                if (info.key === 'logout') {
                  navigate('/login');
                }
              },
            }}
          >
            <Button type="text" icon={<Avatar icon={<UserOutlined />} size="small" />} />
          </Dropdown>

          {/* AI 天按钮 */}
          <Tooltip title="AI Assistant">
            <Button
              type="primary"
              icon={<RobotOutlined />}
              onClick={() => setRightPanelOpen(true)}
            />
          </Tooltip>
        </HeaderRightWrapper>
      </Layout.Header>

      {/* 主内容区 */}
      <Layout.Content style={{ padding: '24px' }}>
        {children}
      </Layout.Content>

      {/* 页脚 */}
      <Layout.Footer>
        <div style={{ textAlign: 'center', padding: '16px', color: 'var(--text-secondary)' }}>
          CodeInsight © 2025 · Version 2.0 · <a href="#">API Documentation</a> · <a href="#">Feedback</a>
        </div>
      </Layout.Footer>
    </LayoutWrapper>

    {/* 右侧面板 */}
    <RightPanel
      open={rightPanelOpen}
      onClose={() => setRightPanelOpen(false)}
      width={320}
      style={{
        position: 'fixed',
        right: 0,
        top: 0,
        bottom: 0,
        height: '100vh',
        overflow: 'auto',
        zIndex: 1000,
      }}
    >
      {/* AI助手内容 */}
      <div style={{ padding: '16px' }}>
        <Title level={4}>AI Assistant</Title>
        <Paragraph>
          Welcome to CodeInsight AI Assistant! I can help you with code analysis, optimization suggestions, and architectural insights.
        </Paragraph>
        
        {/* 快速操作 */}
        <div style={{ marginBottom: '16px' }}>
          <Space direction="vertical" style={{ width: '100%' }}>
            <Button 
              type="primary" 
              block 
              icon={<ThunderboltOutlined />}
              onClick={() => {/* TODO: Implement AI analysis */}}
            >
              Analyze Current Code
            </Button>
            <Button 
              block 
              icon={<BulbOutlined />}
              onClick={() => {/* TODO: Implement suggestions */}}
            >
              Get Suggestions
            </Button>
          </Space>
        </div>

        {/* 分析历史 */}
        <div>
          <Title level={5}>Recent Analysis</Title>
          <List
            size="small"
            dataSource={[
              { title: 'Performance Issue Found', time: '2 min ago' },
              { title: 'Security Vulnerability', time: '5 min ago' },
              { title: 'Code Style Suggestion', time: '10 min ago' },
            ]}
            renderItem={(item) => (
              <List.Item>
                <List.Item.Meta
                  avatar={<Avatar icon={<FileTextOutlined />} size="small" />}
                  title={item.title}
                  description={item.time}
                />
              </List.Item>
            )}
          />
        </div>
      </div>
    </RightPanel>

    {/* 通知面板 */}
    {notificationsOpen && (
      <NotificationPanel
        open={notificationsOpen}
        onClose={() => setNotificationsOpen(false)}
      />
    )}
    </>
  );
};

export default AppShell;