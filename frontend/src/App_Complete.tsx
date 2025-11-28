import React from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { Layout, Typography, Menu, Button } from 'antd';
import AppHeader from './components/AppHeader';
import Dashboard from './pages/Dashboard';

const { Content, Sider } = Layout;
const { Title } = Typography;

function App() {
  const [collapsed, setCollapsed] = React.useState(false);

  const menuItems = [
    {
      key: 'dashboard',
      icon: <span>📊</span>,
      label: 'Dashboard',
    },
    {
      key: 'projects',
      icon: <span>📁</span>,
      label: 'Projects',
    },
    {
      key: 'analysis',
      icon: <span>🔍</span>,
      label: 'Code Analysis',
    },
    {
      key: 'settings',
      icon: <span>⚙️</span>,
      label: 'Settings',
    },
  ];

  return (
    <BrowserRouter>
      <Layout style={{ minHeight: '100vh' }}>
        <AppHeader />
        <Layout>
          <Sider
            collapsible
            collapsed={collapsed}
            onCollapse={value => setCollapsed(value)}
            style={{ background: '#fff' }}
          >
            <Menu
              mode="inline"
              defaultSelectedKeys={['dashboard']}
              style={{ height: '100%', borderRight: 0 }}
              items={menuItems}
              onClick={({ key }) => {
                // Simple navigation - in real app would use navigate
                console.log('Navigate to:', key);
              }}
            />
          </Sider>
          <Layout style={{ padding: '0 24px 24px' }}>
            <Content
              style={{
                padding: 24,
                margin: 0,
                minHeight: 280,
                background: '#fff',
                borderRadius: 8,
              }}
            >
              <Routes>
                <Route path="/" element={<Navigate to="/dashboard" replace />} />
                <Route path="/dashboard" element={<Dashboard />} />
                <Route
                  path="/projects"
                  element={
                    <div>
                      <Title level={2}>Projects</Title>
                      <p>Projects management coming soon...</p>
                      <Button type="primary">New Project</Button>
                    </div>
                  }
                />
                <Route
                  path="/analysis"
                  element={
                    <div>
                      <Title level={2}>Code Analysis</Title>
                      <p>Code analysis tools coming soon...</p>
                      <Button type="primary">Start Analysis</Button>
                    </div>
                  }
                />
                <Route
                  path="/settings"
                  element={
                    <div>
                      <Title level={2}>Settings</Title>
                      <p>Settings panel coming soon...</p>
                    </div>
                  }
                />
              </Routes>
            </Content>
          </Layout>
        </Layout>
      </Layout>
    </BrowserRouter>
  );
}

export default App;
