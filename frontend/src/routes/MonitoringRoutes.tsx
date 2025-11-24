import React from 'react';
import { Route, Routes } from 'react-router-dom';
import MonitoringDashboard from '../components/MonitoringDashboard';

const MonitoringRoutes: React.FC = () => {
  return (
    <Routes>
      <Route path="/" element={<MonitoringDashboard />} />
    </Routes>
  );
};

export default MonitoringRoutes;