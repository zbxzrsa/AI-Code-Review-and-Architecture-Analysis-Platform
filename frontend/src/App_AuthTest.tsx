import React from 'react';
import { Layout, Typography } from 'antd';
import AppHeader from './components/AppHeader';

const { Content } = Layout;
const { Title } = Typography;

function AppWithAuth() {
  return (
    <Layout style={{ minHeight: '100vh' }}>
      <AppHeader />
      <Content style={{ padding: '24px' }}>
        <Title level={1}>AI Code Review Platform</Title>
        <p>Platform with authentication is running!</p>
      </Content>
    </Layout>
  );
}

export default AppWithAuth;
