import React from 'react';
import { Tag } from 'antd';
import { severityColorMap } from '../../styles/tokens';

export interface SeverityTagProps {
  severity?: string;
  children?: React.ReactNode;
}

const SeverityTag: React.FC<SeverityTagProps> = ({ severity = 'INFO', children }) => {
  const key = (severity || 'INFO').toUpperCase() as keyof typeof severityColorMap;
  const map = severityColorMap[key] || severityColorMap.INFO;
  return (
    <Tag style={{ background: map.light, color: map.color, borderRadius: 4 }}>{children || map.label}</Tag>
  );
};

export default SeverityTag;
