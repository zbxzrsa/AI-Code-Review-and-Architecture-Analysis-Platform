import React, { useState, useEffect } from 'react';
import { Card, Row, Col, Tabs, Spin, Alert, Select, Button, Tooltip, Typography } from 'antd';
import { ReloadOutlined, InfoCircleOutlined } from '@ant-design/icons';
import { ResponsiveContainer, LineChart, Line, XAxis, YAxis, Tooltip as RTooltip, CartesianGrid } from 'recharts';
import { performanceMonitor } from '../../utils/performance';
import { Logger } from '../../utils/logger';


const { Title, Text } = Typography;
const { Option } = Select;

// 模拟数据获取函数
const fetchPerformanceData = async (timeRange: string) => {
  // 实际项目中，这里应该从后端API获取数据
  // 这里使用模拟数据
  await new Promise(resolve => setTimeout(resolve, 500));
  
  // 生成随机数据
  const generateTimeData = (count: number, min: number, max: number) => {
    const now = Date.now();
    return Array.from({ length: count }, (_, i) => ({
      timestamp: new Date(now - (count - i) * 60000).toISOString(),
      value: Math.floor(Math.random() * (max - min + 1)) + min
    }));
  };
  
  return {
    pageLoadTime: generateTimeData(20, 500, 3000),
    apiResponseTime: generateTimeData(20, 50, 500),
    resourceLoadTime: generateTimeData(20, 100, 1000),
    firstContentfulPaint: generateTimeData(20, 200, 1500),
    largestContentfulPaint: generateTimeData(20, 500, 2500),
    cumulativeLayoutShift: generateTimeData(20, 0, 0.5).map(item => ({ ...item, value: item.value.toFixed(2) })),
    firstInputDelay: generateTimeData(20, 10, 200),
    memoryUsage: generateTimeData(20, 20, 80),
    cpuUsage: generateTimeData(20, 10, 90),
    errorRate: generateTimeData(20, 0, 5),
    successRate: generateTimeData(20, 90, 100),
  };
};

// 模拟获取健康状态数据
const fetchHealthStatus = async () => {
  // 实际项目中，这里应该从后端API获取数据
  await new Promise(resolve => setTimeout(resolve, 300));
  
  return {
    status: 'healthy', // 'healthy', 'degraded', 'unhealthy'
    dependencies: [
      { name: 'Database', status: 'healthy', responseTime: 15 },
      { name: 'Cache', status: 'healthy', responseTime: 5 },
      { name: 'API Service', status: 'healthy', responseTime: 120 },
      { name: 'Storage', status: 'healthy', responseTime: 80 },
    ],
    uptime: '5d 12h 30m',
    version: '1.0.0',
  };
};

const PerformanceDashboard: React.FC = () => {
  const logger = new Logger('PerformanceDashboard');
  const [timeRange, setTimeRange] = useState<string>('1h');
  const [performanceData, setPerformanceData] = useState<any>(null);
  const [healthData, setHealthData] = useState<any>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  
  // 加载性能数据
  const loadData = async () => {
    try {
      setLoading(true);
      setError(null);
      
      const [perfData, healthStatus] = await Promise.all([
        fetchPerformanceData(timeRange),
        fetchHealthStatus()
      ]);
      
      setPerformanceData(perfData);
      setHealthData(healthStatus);
      
      logger.info('Performance dashboard data loaded', { timeRange });
    } catch (err) {
      setError('Failed to load performance data');
      logger.error('Failed to load performance data', { error: String(err) });
    } finally {
      setLoading(false);
    }
  };
  
  // 初始加载和时间范围变化时重新加载数据
  useEffect(() => {
    loadData();
    
    // 启动性能监控（实际项目中应在应用入口处初始化）
    performanceMonitor.init({
      sampleRate: 1.0, // 仪表板页面100%采样
      enableConsoleLogging: true
    });
    
    return () => {
      // 停止性能监控
      performanceMonitor.stopMonitoring();
    };
  }, [timeRange]);
  
  // 渲染加载状态
  if (loading && !performanceData) {
    return (
      <div style={{ textAlign: 'center', padding: '50px' }}>
        <Spin size="large" />
        <div style={{ marginTop: '20px' }}>Loading performance data...</div>
      </div>
    );
  }
  
  // 渲染错误状态
  if (error) {
    return (
      <Alert
        message="Error"
        description={error}
        type="error"
        showIcon
        action={
          <Button size="small" type="primary" onClick={loadData}>
            Retry
          </Button>
        }
      />
    );
  }
  
  // 配置图表
  const lineConfig = {
    data: [],
    xField: 'timestamp',
    yField: 'value',
    point: {
      size: 3,
      shape: 'circle',
    },
    tooltip: {
      showMarkers: false,
    },
    state: {
      active: {
        style: {
          shadowBlur: 4,
          stroke: '#000',
          fill: 'red',
        },
      },
    },
    interactions: [{ type: 'marker-active' }],
  };
  
  // 健康状态指示器
  const HealthIndicator = ({ status }: { status: string }) => {
    const color = status === 'healthy' ? '#52c41a' : status === 'degraded' ? '#faad14' : '#f5222d';
    return (
      <div style={{ display: 'flex', alignItems: 'center' }}>
        <div style={{ 
          width: 12, 
          height: 12, 
          borderRadius: '50%', 
          backgroundColor: color,
          marginRight: 8
        }} />
        <Text style={{ textTransform: 'capitalize' }}>{status}</Text>
      </div>
    );
  };
  
  return (
    <div className="performance-dashboard">
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 16 }}>
        <Title level={3}>System Performance Dashboard</Title>
        <div>
          <Select 
            value={timeRange} 
            onChange={setTimeRange} 
            style={{ width: 120, marginRight: 16 }}
          >
            <Option value="1h">Last 1 hour</Option>
            <Option value="6h">Last 6 hours</Option>
            <Option value="24h">Last 24 hours</Option>
            <Option value="7d">Last 7 days</Option>
          </Select>
          <Button 
            type="primary" 
            icon={<ReloadOutlined />} 
            onClick={loadData}
          >
            Refresh
          </Button>
        </div>
      </div>
      
      {/* System Health Card */}
      {healthData && (
        <Card 
          title={
            <div style={{ display: 'flex', alignItems: 'center' }}>
              <span>System Health</span>
              <Tooltip title="Shows health of system and dependent services">
                <InfoCircleOutlined style={{ marginLeft: 8 }} />
              </Tooltip>
            </div>
          }
          style={{ marginBottom: 16 }}
        >
          <Row gutter={16}>
            <Col span={6}>
              <Card bordered={false}>
                <Statistic 
                  title="System Status" 
                  value={<HealthIndicator status={healthData.status} />}
                />
              </Card>
            </Col>
            <Col span={6}>
              <Card bordered={false}>
                <Statistic title="Uptime" value={healthData.uptime} />
              </Card>
            </Col>
            <Col span={6}>
              <Card bordered={false}>
                <Statistic title="Version" value={healthData.version} />
              </Card>
            </Col>
            <Col span={6}>
              <Card bordered={false}>
                <Statistic 
                  title="Dependencies" 
                  value={`${healthData.dependencies.filter((d: any) => d.status === 'healthy').length}/${healthData.dependencies.length}`} 
                  suffix="healthy"
                />
              </Card>
            </Col>
          </Row>
          
          <div style={{ marginTop: 16 }}>
            <Title level={5}>Dependency Status</Title>
            <Row gutter={16}>
              {healthData.dependencies.map((dep: any, index: number) => (
                <Col span={6} key={index}>
                  <Card 
                    size="small" 
                    title={dep.name}
                    extra={<HealthIndicator status={dep.status} />}
                  >
                    <div>Response time: {dep.responseTime}ms</div>
                  </Card>
                </Col>
              ))}
            </Row>
          </div>
        </Card>
      )}
      
      {/* Performance Tabs */}
      <Tabs 
        defaultActiveKey="core" 
        type="card"
        items={[
          {
            key: 'core',
            label: 'Core Performance Metrics',
            children: (
              <Row gutter={16}>
                <Col span={12}>
                  <Card 
                    title="Page Load Time (ms)" 
                    extra={<Tooltip title="Time from navigation start to full page load"><InfoCircleOutlined /></Tooltip>}
                  >
                    <ResponsiveContainer width="100%" height={260}>
                      <LineChart data={performanceData?.pageLoadTime || []}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="timestamp" hide />
                        <YAxis domain={[0, 5000]} />
                        <RTooltip />
                        <Line type="monotone" dataKey="value" stroke="#1677ff" dot={false} />
                      </LineChart>
                    </ResponsiveContainer>
                  </Card>
                </Col>
                <Col span={12}>
                  <Card 
                    title="API Response Time (ms)"
                    extra={<Tooltip title="Average response time of API requests"><InfoCircleOutlined /></Tooltip>}
                  >
                    <ResponsiveContainer width="100%" height={260}>
                      <LineChart data={performanceData?.apiResponseTime || []}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="timestamp" hide />
                        <YAxis domain={[0, 1000]} />
                        <RTooltip />
                        <Line type="monotone" dataKey="value" stroke="#52c41a" dot={false} />
                      </LineChart>
                    </ResponsiveContainer>
                  </Card>
                </Col>
              </Row>
            )
          },
          {
            key: 'ux',
            label: 'User Experience Metrics',
            children: (
              <Row gutter={16}>
                <Col span={12}>
                  <Card 
                    title="First Contentful Paint (ms)"
                    extra={<Tooltip title="Time until first content is shown"><InfoCircleOutlined /></Tooltip>}
                  >
                    <ResponsiveContainer width="100%" height={260}>
                      <LineChart data={performanceData?.firstContentfulPaint || []}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="timestamp" hide />
                        <YAxis domain={[0, 2000]} />
                        <RTooltip />
                        <Line type="monotone" dataKey="value" stroke="#1677ff" dot={false} />
                      </LineChart>
                    </ResponsiveContainer>
                  </Card>
                </Col>
                <Col span={12}>
                  <Card 
                    title="Largest Contentful Paint (ms)"
                    extra={<Tooltip title="Time until largest content element is rendered"><InfoCircleOutlined /></Tooltip>}
                  >
                    <ResponsiveContainer width="100%" height={260}>
                      <LineChart data={performanceData?.largestContentfulPaint || []}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="timestamp" hide />
                        <YAxis domain={[0, 3000]} />
                        <RTooltip />
                        <Line type="monotone" dataKey="value" stroke="#52c41a" dot={false} />
                      </LineChart>
                    </ResponsiveContainer>
                  </Card>
                </Col>
              </Row>
            )
          },
          {
            key: 'system',
            label: 'System Resource Metrics',
            children: (
              <Row gutter={16}>
                <Col span={12}>
                  <Card 
                    title="Memory Usage (%)"
                    extra={<Tooltip title="Server memory utilization"><InfoCircleOutlined /></Tooltip>}
                  >
                    <ResponsiveContainer width="100%" height={260}>
                      <LineChart data={performanceData?.memoryUsage || []}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="timestamp" hide />
                        <YAxis domain={[0, 100]} />
                        <RTooltip />
                        <Line type="monotone" dataKey="value" stroke="#1677ff" dot={false} />
                      </LineChart>
                    </ResponsiveContainer>
                  </Card>
                </Col>
                <Col span={12}>
                  <Card 
                    title="CPU Usage (%)"
                    extra={<Tooltip title="Server CPU utilization"><InfoCircleOutlined /></Tooltip>}
                  >
                    <ResponsiveContainer width="100%" height={260}>
                      <LineChart data={performanceData?.cpuUsage || []}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="timestamp" hide />
                        <YAxis domain={[0, 100]} />
                        <RTooltip />
                        <Line type="monotone" dataKey="value" stroke="#52c41a" dot={false} />
                      </LineChart>
                    </ResponsiveContainer>
                  </Card>
                </Col>
              </Row>
            )
          },
          {
            key: 'business',
            label: 'Business Performance Metrics',
            children: (
              <Row gutter={16}>
                <Col span={12}>
                  <Card 
                    title="Operation Success Rate (%)"
                    extra={<Tooltip title="Percentage of successful user operations"><InfoCircleOutlined /></Tooltip>}
                  >
                    <ResponsiveContainer width="100%" height={260}>
                      <LineChart data={performanceData?.operationSuccessRate || []}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="timestamp" hide />
                        <YAxis domain={[80, 100]} />
                        <RTooltip />
                        <Line type="monotone" dataKey="value" stroke="#52c41a" dot={false} />
                      </LineChart>
                    </ResponsiveContainer>
                  </Card>
                </Col>
              </Row>
            )
          }
        ]}
      />
    </div>
  );
};

// 统计数据组件
const Statistic = ({ title, value, suffix }: { title: string, value: any, suffix?: string }) => {
  return (
    <div>
      <div style={{ color: 'rgba(0, 0, 0, 0.45)', fontSize: 14, marginBottom: 4 }}>{title}</div>
      <div style={{ color: 'rgba(0, 0, 0, 0.85)', fontSize: 24, fontWeight: 500 }}>
        {value} {suffix && <span style={{ fontSize: 16 }}>{suffix}</span>}
      </div>
    </div>
  );
};

export default PerformanceDashboard;
