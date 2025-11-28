import React from 'react';
import { Layout, Typography } from 'antd';

const { Content } = Layout;
const { Title } = Typography;

function MinimalApp() {
  return (
    <Layout style={{ minHeight: '100vh', padding: '50px' }}>
      <Content>
        <Title level={1}>AI Code Review Platform - Working!</Title>
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

export default MinimalApp;
