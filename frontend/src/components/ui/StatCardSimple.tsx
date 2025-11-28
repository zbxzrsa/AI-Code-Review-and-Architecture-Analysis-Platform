import React from 'react';
import { Card } from 'antd';

interface StatCardProps {
  title: React.ReactNode;
  value: React.ReactNode;
  subtitle?: React.ReactNode;
}

const StatCard: React.FC<StatCardProps> = ({ title, value, subtitle }) => (
  <Card>
    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
      <div>
        <div style={{ color: '#666', fontSize: '14px' }}>{title}</div>
        <div style={{ fontSize: '24px', fontWeight: 600 }}>{value}</div>
        {subtitle && <div style={{ color: '#666', marginTop: '4px' }}>{subtitle}</div>}
      </div>
    </div>
  </Card>
);

export default StatCard;
