import React from 'react';
import { BrowserRouter, Routes, Route, Navigate, useNavigate } from 'react-router-dom';
import { Layout, Typography, Menu, Button, Row, Col, Card } from 'antd';
import AppHeader from './components/AppHeader';

const { Content, Sider } = Layout;
const { Title } = Typography;

// Simple Dashboard Component
const SimpleDashboard = () => {
  const navigate = useNavigate();

  return (
    <div>
      <Title level={2}>Dashboard</Title>
      <Row gutter={16} style={{ marginBottom: 16 }}>
        <Col xs={24} sm={12} md={8} lg={6}>
          <Card>
            <div style={{ textAlign: 'center' }}>
              <div style={{ fontSize: '24px', fontWeight: 600 }}>42</div>
              <div style={{ color: '#666', marginTop: '4px' }}>Analyzed Projects</div>
            </div>
          </Card>
        </Col>
        <Col xs={24} sm={12} md={8} lg={6}>
          <Card>
            <div style={{ textAlign: 'center' }}>
              <div style={{ fontSize: '24px', fontWeight: 600 }}>156</div>
              <div style={{ color: '#666', marginTop: '4px' }}>Issues Found</div>
            </div>
          </Card>
        </Col>
        <Col xs={24} sm={12} md={8} lg={6}>
          <Card>
            <div style={{ textAlign: 'center' }}>
              <div style={{ fontSize: '24px', fontWeight: 600 }}>8</div>
              <div style={{ color: '#666', marginTop: '4px' }}>Vulnerabilities</div>
            </div>
          </Card>
        </Col>
        <Col xs={24} sm={12} md={8} lg={6}>
          <Card>
            <div style={{ textAlign: 'center' }}>
              <div style={{ fontSize: '24px', fontWeight: 600 }}>78%</div>
              <div style={{ color: '#666', marginTop: '4px' }}>Coverage</div>
            </div>
          </Card>
        </Col>
      </Row>
      <Row gutter={16}>
        <Col xs={24} lg={12}>
          <Card title="Issue Trend">
            <div style={{ fontSize: '18px', fontWeight: 600 }}>28</div>
          </Card>
        </Col>
        <Col xs={24} lg={12}>
          <Card title="Analysis Time">
            <div style={{ fontSize: '18px', fontWeight: 600 }}>3.0m</div>
          </Card>
        </Col>
      </Row>
    </div>
  );
};

function App() {
  const [collapsed, setCollapsed] = React.useState(false);
  const navigate = useNavigate();

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

  const handleMenuClick = ({ key }) => {
    navigate(`/${key}`);
  };

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
            width={200}
          >
            <Menu
              mode="inline"
              defaultSelectedKeys={['dashboard']}
              style={{ height: '100%', borderRight: 0 }}
              items={menuItems}
              onClick={handleMenuClick}
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
                <Route path="/dashboard" element={<SimpleDashboard />} />
                <Route
                  path="/projects"
                  element={
                    <div>
                      <Title level={2}>Projects</Title>
                      <p>Projects management</p>
                      <Button
                        type="primary"
                        onClick={() => alert('New Project clicked!')}
                        style={{ marginTop: 16 }}
                      >
                        New Project
                      </Button>
                    </div>
                  }
                />
                <Route
                  path="/analysis"
                  element={
                    <div>
                      <Title level={2}>Code Analysis</Title>
                      <p>Code analysis tools</p>
                      <Button
                        type="primary"
                        onClick={() => alert('Start Analysis clicked!')}
                        style={{ marginTop: 16 }}
                      >
                        Start Analysis
                      </Button>
                    </div>
                  }
                />
                <Route
                  path="/settings"
                  element={
                    <div>
                      <Title level={2}>Settings</Title>
                      <p>Settings panel</p>
                      <Button
                        type="primary"
                        onClick={() => alert('Settings clicked!')}
                        style={{ marginTop: 16 }}
                      >
                        Save Settings
                      </Button>
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
