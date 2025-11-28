import React from 'react';
import { Routes, Route } from 'react-router-dom';
import { Layout, Typography } from 'antd';
import AppHeader from './components/AppHeader';

const { Content } = Layout;
const { Title } = Typography;

function App() {
  return (
    <Layout style={{ minHeight: '100vh' }}>
      <AppHeader />
      <Content style={{ padding: '24px' }}>
        <Title level={2}>AI Code Review Platform</Title>
        <p>Platform is running successfully!</p>
        <div style={{ marginTop: '20px' }}>
          <a href="http://localhost:8001/health" target="_blank" rel="noopener noreferrer">
            Check Backend Health
          </a>
        </div>
      </Content>
    </Layout>
  );
}

export default App;
