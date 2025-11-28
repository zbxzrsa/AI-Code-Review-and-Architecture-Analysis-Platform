import React from 'react';
import { Layout, Typography, Button, Card } from 'antd';

const { Content, Header } = Layout;
const { Title } = Typography;

function TestApp() {
  const [message, setMessage] = React.useState('Platform loaded!');

  const testClick = () => {
    setMessage('Button clicked! Functions working!');
  };

  const testAPI = async () => {
    try {
      const response = await fetch('http://localhost:8001/health');
      const data = await response.json();
      setMessage(`API test: ${data.status}`);
    } catch (error) {
      setMessage(`API error: ${error.message}`);
    }
  };

  return (
    <Layout style={{ minHeight: '100vh' }}>
      <Header style={{ background: '#fff', padding: '0 24px' }}>
        <Title level={3} style={{ margin: '16px 0' }}>
          AI Code Review Platform
        </Title>
      </Header>
      <Content style={{ padding: '24px' }}>
        <Card title="Platform Status" style={{ marginBottom: 16 }}>
          <p>{message}</p>
          <Space>
            <Button type="primary" onClick={testClick}>
              Test UI
            </Button>
            <Button onClick={testAPI}>Test API</Button>
          </Space>
        </Card>

        <Card title="Quick Links">
          <p>
            <a href="http://localhost:8001/health" target="_blank">
              Backend Health
            </a>
          </p>
          <p>
            <a href="http://localhost:8001/api/v1/auth/login" target="_blank">
              API Login
            </a>
          </p>
        </Card>
      </Content>
    </Layout>
  );
}

export default TestApp;
