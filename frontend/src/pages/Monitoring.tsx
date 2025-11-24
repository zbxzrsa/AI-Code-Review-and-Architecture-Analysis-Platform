import React, { useState, useEffect } from 'react';
import { Card, Row, Col, Typography, Statistic, Alert, Spin, Button, Space, Progress, List, Tag } from 'antd';
import {
  SaveOutlined as ServerOutlined,
  DatabaseOutlined,
  ApiOutlined,
  CheckCircleOutlined,
  ExclamationCircleOutlined,
  ClockCircleOutlined,
  ReloadOutlined
} from '@ant-design/icons';
import { useNavigate } from 'react-router-dom';

const { Title, Text } = Typography;

interface SystemMetrics {
  cpu: number;
  memory: number;
  disk: number;
  uptime: string;
  activeConnections: number;
  apiResponseTime: number;
  errorRate: number;
  lastUpdate: string;
}

interface ServiceStatus {
  name: string;
  status: 'healthy' | 'warning' | 'error';
  uptime: string;
  lastCheck: string;
  responseTime?: number;
}

const Monitoring: React.FC = () => {
  const navigate = useNavigate();
  const [loading, setLoading] = useState(true);
  const [metrics, setMetrics] = useState<SystemMetrics | null>(null);
  const [services, setServices] = useState<ServiceStatus[]>([]);

  useEffect(() => {
    loadMonitoringData();
    const interval = setInterval(loadMonitoringData, 30000); // Refresh every 30 seconds
    return () => clearInterval(interval);
  }, []);

  const loadMonitoringData = async () => {
    try {
      // Simulate API call
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      setMetrics({
        cpu: Math.random() * 100,
        memory: Math.random() * 100,
        disk: Math.random() * 100,
        uptime: '15d 7h 32m',
        activeConnections: Math.floor(Math.random() * 1000),
        apiResponseTime: Math.random() * 500,
        errorRate: Math.random() * 5,
        lastUpdate: new Date().toLocaleTimeString()
      });

      setServices([
        {
          name: 'API Gateway',
          status: Math.random() > 0.2 ? 'healthy' : 'warning',
          uptime: '15d 7h 32m',
          lastCheck: new Date().toLocaleTimeString(),
          responseTime: Math.random() * 200
        },
        {
          name: 'Code Analysis Service',
          status: Math.random() > 0.1 ? 'healthy' : 'error',
          uptime: '15d 7h 32m',
          lastCheck: new Date().toLocaleTimeString(),
          responseTime: Math.random() * 300
        },
        {
          name: 'Database',
          status: 'healthy',
          uptime: '15d 7h 32m',
          lastCheck: new Date().toLocaleTimeString(),
          responseTime: Math.random() * 50
        },
        {
          name: 'Redis Cache',
          status: 'healthy',
          uptime: '15d 7h 32m',
          lastCheck: new Date().toLocaleTimeString(),
          responseTime: Math.random() * 10
        },
        {
          name: 'AI Service',
          status: Math.random() > 0.15 ? 'healthy' : 'warning',
          uptime: '15d 7h 32m',
          lastCheck: new Date().toLocaleTimeString(),
          responseTime: Math.random() * 1000
        }
      ]);

      setLoading(false);
    } catch (error) {
      console.error('Failed to load monitoring data:', error);
      setLoading(false);
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'healthy': return 'green';
      case 'warning': return 'orange';
      case 'error': return 'red';
      default: return 'gray';
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'healthy': return <CheckCircleOutlined style={{ color: '#52c41a' }} />;
      case 'warning': return <ExclamationCircleOutlined style={{ color: '#faad14' }} />;
      case 'error': return <ExclamationCircleOutlined style={{ color: '#ff4d4f' }} />;
      default: return <ClockCircleOutlined style={{ color: '#d9d9d9' }} />;
    }
  };

  if (loading) {
    return (
      <div style={{ padding: '24px', textAlign: 'center' }}>
        <Spin size="large" />
        <Title level={4} style={{ marginTop: 16 }}>Loading monitoring data...</Title>
      </div>
    );
  }

  return (
    <div style={{ padding: '24px' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '24px' }}>
        <Title level={2}>System Monitoring</Title>
        <Space>
          <Button icon={<ReloadOutlined />} onClick={loadMonitoringData}>
            Refresh
          </Button>
          <Button type="primary" onClick={() => navigate('/settings')}>
            Configure Alerts
          </Button>
        </Space>
      </div>

      <Text type="secondary">Last updated: {metrics?.lastUpdate}</Text>

      {/* System Metrics */}
      <Row gutter={[16, 16]} style={{ marginTop: 24 }}>
        <Col xs={24} sm={12} md={8} lg={6}>
          <Card>
            <Statistic
              title="CPU Usage"
              value={metrics?.cpu.toFixed(1)}
              suffix="%"
              prefix={<ServerOutlined />}
              valueStyle={{ color: metrics && metrics.cpu > 80 ? '#ff4d4f' : '#3f8600' }}
            />
            <Progress 
              percent={metrics?.cpu} 
              size="small" 
              status={metrics && metrics.cpu > 80 ? 'exception' : 'normal'}
              showInfo={false}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} md={8} lg={6}>
          <Card>
            <Statistic
              title="Memory Usage"
              value={metrics?.memory.toFixed(1)}
              suffix="%"
              prefix={<DatabaseOutlined />}
              valueStyle={{ color: metrics && metrics.memory > 85 ? '#ff4d4f' : '#3f8600' }}
            />
            <Progress 
              percent={metrics?.memory} 
              size="small" 
              status={metrics && metrics.memory > 85 ? 'exception' : 'normal'}
              showInfo={false}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} md={8} lg={6}>
          <Card>
            <Statistic
              title="Disk Usage"
              value={metrics?.disk.toFixed(1)}
              suffix="%"
              prefix={<DatabaseOutlined />}
              valueStyle={{ color: metrics && metrics.disk > 90 ? '#ff4d4f' : '#3f8600' }}
            />
            <Progress 
              percent={metrics?.disk} 
              size="small" 
              status={metrics && metrics.disk > 90 ? 'exception' : 'normal'}
              showInfo={false}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} md={8} lg={6}>
          <Card>
            <Statistic
              title="API Response Time"
              value={metrics?.apiResponseTime.toFixed(0)}
              suffix="ms"
              prefix={<ApiOutlined />}
              valueStyle={{ color: metrics && metrics.apiResponseTime > 500 ? '#ff4d4f' : '#3f8600' }}
            />
          </Card>
        </Col>
      </Row>

      {/* Additional Metrics */}
      <Row gutter={[16, 16]} style={{ marginTop: 16 }}>
        <Col xs={24} sm={12} md={8}>
          <Card>
            <Statistic
              title="Active Connections"
              value={metrics?.activeConnections}
              prefix={<ApiOutlined />}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} md={8}>
          <Card>
            <Statistic
              title="Error Rate"
              value={metrics?.errorRate.toFixed(2)}
              suffix="%"
              prefix={<ExclamationCircleOutlined />}
              valueStyle={{ color: metrics && metrics.errorRate > 1 ? '#ff4d4f' : '#3f8600' }}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} md={8}>
          <Card>
            <Statistic
              title="System Uptime"
              value={metrics?.uptime}
              prefix={<ClockCircleOutlined />}
            />
          </Card>
        </Col>
      </Row>

      {/* Service Status */}
      <Card title="Service Status" style={{ marginTop: 24 }}>
        <List
          dataSource={services}
          renderItem={(service) => (
            <List.Item>
              <List.Item.Meta
                avatar={getStatusIcon(service.status)}
                title={
                  <Space>
                    {service.name}
                    <Tag color={getStatusColor(service.status)}>{service.status}</Tag>
                  </Space>
                }
                description={
                  <Space>
                    <Text type="secondary">Uptime: {service.uptime}</Text>
                    {service.responseTime && (
                      <Text type="secondary">Response: {service.responseTime.toFixed(0)}ms</Text>
                    )}
                    <Text type="secondary">Last check: {service.lastCheck}</Text>
                  </Space>
                }
              />
            </List.Item>
          )}
        />
      </Card>

      {/* Alerts */}
      {services.some(s => s.status !== 'healthy') && (
        <Alert
          message="Service Issues Detected"
          description="Some services are experiencing issues. Please check the service status above and take appropriate action."
          type="warning"
          showIcon
          style={{ marginTop: 24 }}
        />
      )}
    </div>
  );
};

export default Monitoring;