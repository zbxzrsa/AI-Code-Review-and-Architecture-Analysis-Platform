import React, { useState } from 'react';
import { Card, Typography, Input, Button, Space, Segmented, DatePicker, Select, Table, Tag, Row, Col, Statistic, List, Alert } from 'antd';
 
import type { ColumnsType } from 'antd/es/table';

interface SearchParams {
  keyword: string;
  type: 'project' | 'session' | 'file' | 'issue' | 'analysis';
  project?: string;
  session?: string;
  baseName?: string;
  startDate?: string;
  endDate?: string;
  page?: number;
  pageSize?: number;
}

interface SearchResult {
  id: string;
  title: string;
  type: 'project' | 'session' | 'file' | 'issue' | 'analysis';
  project?: string;
  session?: string;
  file?: string;
  content?: string;
  relevance?: number;
  createdTime?: string;
}

const Search: React.FC = () => {
  const t = (k: string, fb?: string) => fb || k;
  const [params, setParams] = useState<SearchParams>({
    keyword: '',
    type: 'project',
    project: '',
    session: '',
    baseName: '',
    startDate: '',
    endDate: '',
    page: 1,
    pageSize: 10
  });
  
  const [results, setResults] = useState<SearchResult[]>([]);
  const [loading, setLoading] = useState(false);
  const [showFilters, setShowFilters] = useState(false);
  const [aiSummary, setAiSummary] = useState<string>('');
  
  const handleSearch = async () => {
    setLoading(true);
    // Simulated result set for the demo environment
    setTimeout(() => {
      const mock: SearchResult[] = [
        {
          id: 'proj-001',
          title: 'E-Commerce Experience',
          type: 'project',
          project: 'E-Commerce Experience',
          content: 'Quality score improved by +6 after refactor.',
          relevance: 0.93,
          createdTime: '2024-11-05T13:00:00Z',
        },
        {
          id: 'session-202',
          title: 'Security hardening sprint',
          type: 'session',
          session: 'Security hardening sprint',
          content: 'Dynamic analysis flagged XSS regression.',
          relevance: 0.88,
          createdTime: '2024-11-04T10:00:00Z',
        },
        {
          id: 'file-123',
          title: 'src/services/order.ts',
          type: 'file',
          file: 'src/services/order.ts',
          content: 'Removed anti-pattern by introducing repository class.',
          relevance: 0.82,
          createdTime: '2024-11-03T08:00:00Z',
        },
      ];
      setResults(mock);
      setAiSummary(
        'Top matches relate to the commerce platform refactor and a recent security sprint. ' +
          'Quality metrics trend upward while one session still needs attention.'
      );
      setLoading(false);
    }, 600);
  };
  
  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
    const { name, value } = e.target;
    setParams(prev => ({ ...prev, [name]: value }));
  };
  
  const handleTypeChange = (type: 'project' | 'session' | 'file' | 'issue' | 'analysis') => {
    setParams(prev => ({ ...prev, type }));
  };
  
  const toggleFilters = () => setShowFilters(!showFilters);
  
  const applyFilters = () => {
    setParams(prev => ({ ...prev, page: 1 }));
    handleSearch();
  };
  
  const resetFilters = () => {
    setParams({
      keyword: '',
      type: 'project',
      project: '',
      session: '',
      baseName: '',
      startDate: '',
      endDate: '',
      page: 1,
      pageSize: 10
    });
    setResults([]);
  };

  const columns: ColumnsType<SearchResult> = [
    { title: t('search.result_type'), dataIndex: 'type', key: 'type', render: (type) => <Tag>{t(`search.type.${type}`)}</Tag> },
    { title: t('general.details', 'Details'), dataIndex: 'title', key: 'title' },
    { title: t('search.created_time'), dataIndex: 'createdTime', key: 'createdTime' },
  ];

  return (
    <Space direction="vertical" size="large" style={{ width: '100%' }}>
      <Card>
        <Typography.Title level={3}>{t('search.web_search', 'Unified search')}</Typography.Title>
        <Space wrap style={{ marginBottom: 16 }}>
          <Input
            placeholder={t('search.placeholder', 'Search across projects, sessions, files...')}
            value={params.keyword}
            onChange={(e) => setParams(prev => ({ ...prev, keyword: e.target.value }))}
            style={{ width: 320 }}
          />
          <Segmented
            value={params.type}
            onChange={(v) => handleTypeChange(v as any)}
            options={['project', 'session', 'file', 'issue']}
          />
          <Button type="default" onClick={toggleFilters}>
            {showFilters ? t('search.hide_filters', 'Hide filters') : t('search.show_filters', 'Advanced filters')}
          </Button>
          <Button type="primary" onClick={handleSearch} loading={loading}>
            {t('search.search', 'Search')}
          </Button>
        </Space>

        {showFilters && (
          <Space wrap style={{ marginBottom: 16 }}>
            <Select
              placeholder="Project"
              style={{ width: 200 }}
              value={params.project}
              onChange={(v) => setParams(prev => ({ ...prev, project: v }))}
              allowClear
              options={[{ value: 'proj-001', label: 'E-Commerce Experience' }, { value: 'proj-002', label: 'Observability Gateway' }]}
            />
            <Select
              placeholder="Session"
              style={{ width: 200 }}
              value={params.session}
              onChange={(v) => setParams(prev => ({ ...prev, session: v }))}
              allowClear
              options={[{ value: 'session1', label: 'Regression Sprint' }, { value: 'session2', label: 'Security Scan' }]}
            />
            <DatePicker placeholder="Start date" onChange={(d) => setParams(prev => ({ ...prev, startDate: d?.toISOString() || '' }))} />
            <DatePicker placeholder="End date" onChange={(d) => setParams(prev => ({ ...prev, endDate: d?.toISOString() || '' }))} />
            <Button onClick={applyFilters}>Apply filters</Button>
            <Button onClick={resetFilters}>Reset</Button>
          </Space>
        )}
      </Card>

      {results.length > 0 && (
        <Row gutter={16}>
          <Col span={8}>
            <Card>
              <Statistic title="Projects" value={results.filter(r => r.type === 'project').length} />
            </Card>
          </Col>
          <Col span={8}>
            <Card>
              <Statistic title="Sessions" value={results.filter(r => r.type === 'session').length} />
            </Card>
          </Col>
          <Col span={8}>
            <Card>
              <Statistic title="Files" value={results.filter(r => r.type === 'file').length} />
            </Card>
          </Col>
        </Row>
      )}

      {aiSummary && (
        <Card>
          <Typography.Title level={4}>AI summary</Typography.Title>
          <Typography.Paragraph>{aiSummary}</Typography.Paragraph>
        </Card>
      )}

      <Card>
        {results.length === 0 ? (
          <Typography.Paragraph type="secondary">
            {params.keyword ? t('search.no_results', 'No results yet') : t('search.enter_keywords', 'Enter a query to get started')}
          </Typography.Paragraph>
        ) : (
          <Table columns={columns} dataSource={results} rowKey="id" pagination={{ pageSize: params.pageSize }} />
        )}
      </Card>

      <Card title="Example demonstrations">
        <List
          dataSource={[
            'Filter by project and time range to find drifting baselines.',
            'Search for session labels to resume paused analyses.',
            'Locate files that contain a keyword before generating a diff.',
          ]}
          renderItem={(item) => (
            <List.Item>
              <Alert message={item} type="info" showIcon />
            </List.Item>
          )}
        />
      </Card>
    </Space>
  );
};

export default Search;
