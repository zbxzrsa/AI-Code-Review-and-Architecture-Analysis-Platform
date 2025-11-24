import React, { useEffect, useState } from 'react';
import { Row, Col, Typography, Spin, Alert } from 'antd';
import StatCard from '../components/ui/StatCard';
import TrendCard from '../components/ui/TrendCard';
import { fetchMetrics, Metrics } from '../services/metricsService';

const Dashboard: React.FC = () => {
  const [metrics, setMetrics] = useState<Metrics | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let mounted = true;
    setLoading(true);
    setError(null);

    fetchMetrics(2)
      .then((data: Metrics) => {
        if (!mounted) return;
        setMetrics(data);
      })
      .catch((err: unknown) => {
        if (!mounted) return;
        console.error('Failed to load metrics', err);
        setError((err as Error).message || 'Failed to load dashboard metrics');
      })
      .finally(() => mounted && setLoading(false));

    return () => {
      mounted = false;
    };
  }, []);

  return (
    <div>
      <Typography.Title level={2}>Dashboard</Typography.Title>

      {error && (
        <Alert
          message="无法加载数据"
          description={error}
          type="error"
          closable
          style={{ marginBottom: 16 }}
        />
      )}

      {loading && (
        <div style={{ padding: 24, display: 'flex', justifyContent: 'center' }}>
          <Spin tip="加载数据中..." />
        </div>
      )}

      {!loading && metrics && (
        <>
          <Row gutter={16} style={{ marginBottom: 16 }}>
            <Col xs={24} sm={12} md={8} lg={6}>
              <StatCard title="Analyzed Projects" value={metrics.kpis.analyzedProjects} subtitle="最近 7 天" />
            </Col>
            <Col xs={24} sm={12} md={8} lg={6}>
              <StatCard title="Issues Found" value={metrics.kpis.issuesFound} subtitle="高风险 6" />
            </Col>
            <Col xs={24} sm={12} md={8} lg={6}>
              <StatCard title="Security Vulnerabilities" value={metrics.kpis.vulnerabilities} subtitle="修复优先级：高" />
            </Col>
            <Col xs={24} sm={12} md={8} lg={6}>
              <StatCard title="Coverage" value={`${metrics.kpis.coverage}%`} subtitle="测试覆盖率" />
            </Col>
          </Row>

          <Row gutter={16}>
            <Col xs={24} lg={12} style={{ marginBottom: 16 }}>
              <TrendCard title="Issue Trend" value={`${metrics.trends.issueTrend.slice(-1)[0]}`} sparkData={metrics.trends.issueTrend} />
            </Col>
            <Col xs={24} lg={12} style={{ marginBottom: 16 }}>
              <TrendCard title="Analysis Time" value={`${metrics.trends.analysisTime.slice(-1)[0]}m`} sparkData={metrics.trends.analysisTime} />
            </Col>
          </Row>
        </>
      )}
    </div>
  );
};

export default Dashboard;
