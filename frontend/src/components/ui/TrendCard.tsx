import React from 'react';
import { Card } from 'antd';

interface TrendCardProps {
  title: React.ReactNode;
  value: React.ReactNode;
  sparkData?: number[];
}

const TrendCard: React.FC<TrendCardProps> = ({ title, value }) => {
  return (
    <Card>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div>
          <div style={{ color: '#666', fontSize: '12px' }}>{title}</div>
          <div style={{ fontSize: '24px', fontWeight: 600 }}>{value}</div>
        </div>
        <div
          style={{
            width: 120,
            height: 40,
            background: 'linear-gradient(90deg, rgba(46,144,250,0.08), transparent)',
            borderRadius: '4px',
          }}
        />
      </div>
    </Card>
  );
};

export default TrendCard;
