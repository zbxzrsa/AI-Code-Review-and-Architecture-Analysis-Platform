import React from 'react';
import { Card, Typography } from 'antd';

const { Title, Paragraph } = Typography;

const MonitoringDashboard: React.FC = () => {
  return (
    <Card>
      <Title level={3}>Monitoring Dashboard</Title>
      <Paragraph type="secondary">Work in progress</Paragraph>
    </Card>
  );
};

export default MonitoringDashboard;
