import React, { useCallback, useEffect, useMemo, useState } from 'react';
import {
  Table,
  Button,
  Space,
  Card,
  Tag,
  Collapse,
  Tabs,
  Input,
  List,
  Spin,
  Empty,
  Row,
  Col,
  Statistic,
  Segmented,
  Tooltip,
  Badge,
  Divider,
  Typography,
} from 'antd';
import {
  PlusOutlined,
  FolderOpenOutlined,
  FileOutlined,
  ThunderboltOutlined,
  AlertOutlined,
  DeploymentUnitOutlined,
} from '@ant-design/icons';
import { useNavigate } from 'react-router-dom';
import ApiService from '../services/api';
import {
  ProjectRecord,
  projectRepository,
} from '../services/projectRepository';

const api = new ApiService();
const { Title, Paragraph } = Typography;

interface Project {
  id: string;
  name: string;
  description?: string;
  created_at?: string;
}

interface ProjectFile {
  path: string;
  version_count: number;
}

const EXT_LANGUAGE_MAP: Record<string, string> = {
  '.ts': 'TypeScript', '.tsx': 'TypeScript', '.js': 'JavaScript', '.jsx': 'JavaScript',
  '.py': 'Python', '.java': 'Java', '.go': 'Go', '.cs': 'C#', '.cpp': 'C++', '.c': 'C',
  '.rb': 'Ruby', '.php': 'PHP', '.html': 'HTML', '.css': 'CSS', '.scss': 'CSS', '.json': 'JSON',
  '.md': 'Markdown'
};

const splitPath = (p: string): string[] => p.split(/[\\/]+/).filter(Boolean);
const extOf = (p: string): string => {
  const m = p.match(/\.[^.\\/]+$/);
  return m ? m[0].toLowerCase() : '';
};

const FilesPanel: React.FC<{ projectId: string }> = ({ projectId }) => {
  const [files, setFiles] = useState<ProjectFile[]>([]);
  const [loading, setLoading] = useState(false);
  const [query, setQuery] = useState('');

  const fmt = (s: string, params: Record<string, any>) => s.replace(/\{\{(\w+)\}\}/g, (_, k) => String(params[k] ?? ''));

  useEffect(() => {
    const fetchFiles = async () => {
      setLoading(true);
      const res = await api.getProjectFiles(projectId);
      if (res.success && res.data) {
        setFiles(res.data);
      }
      setLoading(false);
    };
    fetchFiles();
  }, [projectId]);

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return files;
    return files.filter(f => f.path.toLowerCase().includes(q));
  }, [files, query]);

  const byLanguage = useMemo(() => {
    const groups: Record<string, ProjectFile[]> = {};
    for (const f of filtered) {
      const lang = EXT_LANGUAGE_MAP[extOf(f.path)] || 'Other';
      groups[lang] = groups[lang] || [];
      groups[lang].push(f);
    }
    return groups;
  }, [filtered]);

  const byDirectory = useMemo(() => {
    const groups: Record<string, ProjectFile[]> = {};
    for (const f of filtered) {
      const parts = splitPath(f.path);
      const dir = parts.length > 1 ? parts[0] : 'Root';
      groups[dir] = groups[dir] || [];
      groups[dir].push(f);
    }
    return groups;
  }, [filtered]);

  const renderGroupList = (items: ProjectFile[]) => (
    <List
      itemLayout="horizontal"
      dataSource={items}
      renderItem={(item) => (
        <List.Item>
          <List.Item.Meta
            avatar={<FileOutlined />}
            title={item.path}
            description={fmt('Versions: {{count}}', { count: item.version_count })}
          />
        </List.Item>
      )}
    />
  );

  return (
    <Card size="small" style={{ marginTop: 8 }}>
      <Space style={{ marginBottom: 12 }} wrap>
        <Input.Search
          allowClear
          placeholder="Search project files..."
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          style={{ width: 320 }}
        />
      </Space>

        {loading ? (
          <div style={{ textAlign: 'center', padding: 24 }}>
            <Spin tip="Loading..." />
          </div>
        ) : filtered.length === 0 ? (
          <Empty description="No files" />
      ) : (
        <Tabs
          defaultActiveKey="language"
          items={[
            {
              key: 'language',
              label: 'Group by language',
              children: (
                <Collapse bordered={false}>
                  {Object.entries(byLanguage).map(([lang, items]) => (
                    <Collapse.Panel key={lang} header={
                      <Space>
                        <Tag color="blue">{lang}</Tag>
                        <span>{fmt('Files: {{count}}', { count: items.length })}</span>
                      </Space>
                    }>
                      {renderGroupList(items)}
                    </Collapse.Panel>
                  ))}
                </Collapse>
              )
            },
            {
              key: 'directory',
              label: 'Group by directory',
              children: (
                <Collapse bordered={false}>
                  {Object.entries(byDirectory).map(([dir, items]) => (
                    <Collapse.Panel key={dir} header={
                      <Space>
                        <FolderOpenOutlined />
                        <span>{dir}</span>
                        <Tag>{fmt('Files: {{count}}', { count: items.length })}</Tag>
                      </Space>
                    }>
                      {renderGroupList(items)}
                    </Collapse.Panel>
                  ))}
                </Collapse>
              )
            }
          ]}
        />
      )}
    </Card>
  );
};

const Projects: React.FC = () => {
  const navigate = useNavigate();
  const [projects, setProjects] = useState<ProjectRecord[]>([]);
  const [loading, setLoading] = useState(false);
  const [searchTerm, setSearchTerm] = useState('');
  const [statusFilter, setStatusFilter] = useState<string>('all');
  const [qualityFilter, setQualityFilter] = useState<string>('all');

  const loadProjects = useCallback(async () => {
    setLoading(true);
    try {
      const remote = await api.getProjects();
      if (remote.success && remote.data?.length) {
        setProjects(
          remote.data.map((proj) => ({
            id: proj.id,
            name: proj.name,
            description: proj.description,
            status: 'Active',
            owner: 'API Owner',
            tags: [],
            repoUrl: undefined,
            lastAnalysis: proj.updated_at,
            qualityScore: 75,
            riskScore: 25,
            createdAt: proj.created_at,
            updatedAt: proj.updated_at,
          }))
        );
      } else {
        setProjects(projectRepository.list());
      }
    } catch {
      setProjects(projectRepository.list());
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadProjects();
  }, [loadProjects]);

  const filteredProjects = useMemo(() => {
    return projects.filter((project) => {
      const matchesSearch =
        project.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
        project.description.toLowerCase().includes(searchTerm.toLowerCase());
      const matchesStatus =
        statusFilter === 'all' ? true : project.status === statusFilter;
      const matchesQuality =
        qualityFilter === 'all'
          ? true
          : qualityFilter === 'excellent'
          ? project.qualityScore >= 85
          : qualityFilter === 'warning'
          ? project.qualityScore < 60
          : project.qualityScore >= 60 && project.qualityScore < 85;
      return matchesSearch && matchesStatus && matchesQuality;
    });
  }, [projects, searchTerm, statusFilter, qualityFilter]);

  const summary = useMemo(() => {
    const total = projects.length;
    const averageQuality =
      total === 0
        ? 0
        : Math.round(
            projects.reduce((acc, curr) => acc + curr.qualityScore, 0) / total
          );
    const openRisks = projects.filter((project) => project.riskScore > 25).length;
    const activeSessions = Math.max(1, Math.round(total * 0.6));
    return { total, averageQuality, openRisks, activeSessions };
  }, [projects]);

  const columns = [
    {
      title: 'Project',
      dataIndex: 'name',
      key: 'name',
      render: (_: unknown, record: ProjectRecord) => (
        <Space direction="vertical" size={0}>
          <Space>
            <Button
              type="link"
              onClick={() => navigate(`/projects/${record.id}`)}
              style={{ padding: 0 }}
            >
              {record.name}
            </Button>
            <Badge
              status={record.status === 'Active' ? 'success' : 'default'}
              text={record.status}
            />
          </Space>
          <span style={{ color: '#6c757d' }}>{record.description}</span>
        </Space>
      ),
    },
    {
      title: 'Owner',
      dataIndex: 'owner',
      key: 'owner',
    },
    {
      title: 'Quality',
      dataIndex: 'qualityScore',
      key: 'quality',
      render: (score: number) => (
        <Tag color={score >= 85 ? 'green' : score >= 60 ? 'gold' : 'red'}>
          {score} / 100
        </Tag>
      ),
    },
    {
      title: 'Risk',
      dataIndex: 'riskScore',
      key: 'risk',
      render: (score: number) => (
        <Tag color={score > 30 ? 'red' : score > 15 ? 'orange' : 'green'}>
          {score}%
        </Tag>
      ),
    },
    {
      title: 'Updated',
      dataIndex: 'updatedAt',
      key: 'updatedAt',
      render: (date: string) => new Date(date).toLocaleString(),
    },
    {
      title: 'Actions',
      key: 'actions',
      render: (_: unknown, record: ProjectRecord) => (
        <Space>
          <Button
            type="link"
            onClick={() => navigate(`/projects/${record.id}`)}
          >
            View
          </Button>
          <Button
            type="link"
            onClick={() => navigate(`/projects/${record.id}/edit`)}
          >
            Edit
          </Button>
          <Button
            type="link"
            onClick={() =>
              navigate(`/analysis?projectId=${encodeURIComponent(record.id)}`)
            }
          >
            Analyze
          </Button>
          <Button
            type="link"
            danger
            onClick={() => navigate(`/projects/${record.id}/danger-zone`)}
          >
            Delete
          </Button>
        </Space>
      ),
    },
  ];

  const exampleScenarios = [
    {
      title: 'Create a dedicated architecture stream',
      content:
        'Capture owner, environments, GitHub repository, and AI guardrails before kicking off the first session.',
      action: 'Launch Create Project wizard',
      icon: <PlusOutlined />,
      onAction: () => navigate('/projects/new'),
    },
    {
      title: 'Promote a branch to Baseline 2.0',
      content:
        'Open the project detail, review baseline drift, and publish the new release candidate to the baseline module.',
      action: 'Open Baseline comparison',
      icon: <DeploymentUnitOutlined />,
      onAction: () => navigate('/baselines'),
    },
    {
      title: 'Investigate a regression',
      content:
        'Filter projects by risk, open the latest session, and deep-dive into diff visualizations from the Versions module.',
      action: 'Jump to Version Intelligence',
      icon: <ThunderboltOutlined />,
      onAction: () => navigate('/versions'),
    },
  ];

  return (
    <div>
      <div
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          marginBottom: 16,
        }}
      >
        <h1>Projects</h1>
        <Space>
          <Button onClick={() => navigate('/projects/new')} icon={<PlusOutlined />}>
            New Project
          </Button>
          <Button type="primary" onClick={() => navigate('/projects/import')}>
            Import Repository
          </Button>
        </Space>
      </div>

      <Row gutter={16}>
        <Col span={6}>
          <Card>
            <Statistic title="Total Projects" value={summary.total} />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="Average Quality Score"
              value={summary.averageQuality}
              suffix="/ 100"
              valueStyle={{ color: '#52c41a' }}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="Open Risk Streams"
              value={summary.openRisks}
              prefix={<AlertOutlined />}
              valueStyle={{ color: '#faad14' }}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="Active Sessions"
              value={summary.activeSessions}
              prefix={<ThunderboltOutlined />}
            />
          </Card>
        </Col>
      </Row>

      <Row gutter={16} style={{ marginTop: 16 }}>
        <Col span={8}>
          <Card title="Example demonstrations">
            <List
              itemLayout="vertical"
              dataSource={exampleScenarios}
              renderItem={(item) => (
                <List.Item
                  actions={[
                    <Button
                      key={item.title}
                      size="small"
                      type="link"
                      icon={item.icon}
                      onClick={item.onAction}
                    >
                      {item.action}
                    </Button>,
                  ]}
                >
                  <List.Item.Meta title={item.title} />
                  <Paragraph style={{ marginBottom: 0 }}>{item.content}</Paragraph>
                </List.Item>
              )}
            />
          </Card>
        </Col>
        <Col span={16}>
          <Card
            title="Project portfolio"
            extra={
              <Space>
                <Input.Search
                  placeholder="Search by name or description"
                  style={{ width: 220 }}
                  allowClear
                  value={searchTerm}
                  onChange={(event) => setSearchTerm(event.target.value)}
                />
                <Segmented
                  options={[
                    { label: 'All', value: 'all' },
                    { label: 'Excellent', value: 'excellent' },
                    { label: 'Stable', value: 'stable' },
                    { label: 'Needs attention', value: 'warning' },
                  ]}
                  value={qualityFilter}
                  onChange={(value) => setQualityFilter(value as string)}
                />
                <Segmented
                  options={[
                    { label: 'Status: All', value: 'all' },
                    { label: 'Active', value: 'Active' },
                    { label: 'On Hold', value: 'On Hold' },
                    { label: 'Archived', value: 'Archived' },
                  ]}
                  value={statusFilter}
                  onChange={(value) => setStatusFilter(value as string)}
                />
              </Space>
            }
          >
            <Table
              rowKey={(p: ProjectRecord) => p.id}
              columns={columns as any}
              dataSource={filteredProjects}
              loading={loading}
              pagination={{
                showSizeChanger: true,
                showQuickJumper: true,
                showTotal: (total) => `Total ${total} projects`,
              }}
              expandable={{
                expandedRowRender: (record: ProjectRecord) => (
                  <div>
                    <FilesPanel projectId={record.id} />
                    <Divider />
                    <Title level={5}>Tags</Title>
                    <Space wrap>
                      {record.tags.map((tag) => (
                        <Tag key={tag}>{tag}</Tag>
                      ))}
                    </Space>
                  </div>
                ),
              }}
            />
          </Card>
        </Col>
      </Row>
    </div>
  );
};

export default Projects;