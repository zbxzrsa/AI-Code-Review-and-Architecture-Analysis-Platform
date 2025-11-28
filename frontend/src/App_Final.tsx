import React from 'react';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { Layout, Typography, Row, Col, Card } from 'antd';
import AppHeader from './components/AppHeader';

const { Content } = Layout;
const { Title } = Typography;

function App() {
  return (
    <BrowserRouter>
      <Layout style={{ minHeight: '100vh' }}>
        <AppHeader />
        <Content style={{ padding: '24px' }}>
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
        </Content>
      </Layout>
    </BrowserRouter>
  );
}

export default App;
