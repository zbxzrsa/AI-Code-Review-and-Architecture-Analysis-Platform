import React, { useMemo, useState } from 'react';
import {
  Card,
  Typography,
  List,
  Tag,
  Space,
  Button,
  Input,
  Segmented,
  Statistic,
  Row,
  Col,
} from 'antd';
import { useNavigate } from 'react-router-dom';
import {
  ProjectRecord,
  projectRepository,
} from '../services/projectRepository';

const { Title, Paragraph, Text } = Typography;

const ProjectList: React.FC = () => {
  const navigate = useNavigate();
  const [searchTerm, setSearchTerm] = useState('');
  const [statusFilter, setStatusFilter] = useState('all');
  const projects = projectRepository.list();

  const filteredProjects = useMemo(() => {
    return projects.filter((project) => {
      const matchesSearch =
        project.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
        project.description.toLowerCase().includes(searchTerm.toLowerCase());
      const matchesStatus =
        statusFilter === 'all' ? true : project.status === statusFilter;
      return matchesSearch && matchesStatus;
    });
  }, [projects, searchTerm, statusFilter]);

  return (
    <Space direction="vertical" size="large" style={{ width: '100%' }}>
      <Row gutter={16}>
        <Col span={8}>
          <Card>
            <Statistic title="Projects" value={projects.length} />
          </Card>
        </Col>
        <Col span={8}>
          <Card>
            <Statistic
              title="Active initiatives"
              value={projects.filter((project) => project.status === 'Active').length}
              valueStyle={{ color: '#52c41a' }}
            />
          </Card>
        </Col>
        <Col span={8}>
          <Card>
            <Statistic
              title="Average score"
              suffix="/ 100"
              value={
                projects.length === 0
                  ? 0
                  : Math.round(
                      projects.reduce((sum, project) => sum + project.qualityScore, 0) /
                        projects.length
                    )
              }
            />
          </Card>
        </Col>
      </Row>

      <Card
        title="Project directory"
        extra={
          <Space>
            <Input.Search
              placeholder="Search projects"
              allowClear
              value={searchTerm}
              onChange={(event) => setSearchTerm(event.target.value)}
            />
            <Segmented
              options={[
                { label: 'All', value: 'all' },
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
        <List
          dataSource={filteredProjects}
          itemLayout="vertical"
          renderItem={(project: ProjectRecord) => (
            <List.Item
              key={project.id}
              actions={[
                <Button
                  key="view"
                  type="link"
                  onClick={() => navigate(`/projects/${project.id}`)}
                >
                  View details
                </Button>,
                <Button
                  key="edit"
                  type="link"
                  onClick={() => navigate(`/projects/${project.id}/edit`)}
                >
                  Edit
                </Button>,
                <Button
                  key="sessions"
                  type="link"
                  onClick={() =>
                    navigate(`/sessions?projectId=${encodeURIComponent(project.id)}`)
                  }
                >
                  Sessions
                </Button>,
              ]}
            >
              <List.Item.Meta
                title={
                  <Space>
                    <Text strong>{project.name}</Text>
                    <Tag color={project.status === 'Active' ? 'green' : 'default'}>
                      {project.status}
                    </Tag>
                  </Space>
                }
                description={
                  <Paragraph style={{ marginBottom: 8 }}>{project.description}</Paragraph>
                }
              />
              <Space wrap>
                {project.tags.map((tag) => (
                  <Tag key={tag}>{tag}</Tag>
                ))}
              </Space>
            </List.Item>
          )}
        />
      </Card>

      <Card title="Demonstrations">
        <List
          grid={{ gutter: 16, column: 3 }}
          dataSource={[
            {
              title: 'Create new project',
              description: 'Walk through the guided wizard to set ownership and guardrails.',
              action: () => navigate('/projects/new'),
            },
            {
              title: 'Import from GitHub',
              description: 'Connect through OAuth and sync metadata automatically.',
              action: () => navigate('/github-connect'),
            },
            {
              title: 'Clean up archived projects',
              description: 'Use the danger zone view to retire unused workstreams.',
              action: () => navigate('/projects/archive'),
            },
          ]}
          renderItem={(item) => (
            <List.Item>
              <Card
                title={item.title}
                actions={[
                  <Button type="link" key={item.title} onClick={item.action}>
                    Launch demo
                  </Button>,
                ]}
              >
                {item.description}
              </Card>
            </List.Item>
          )}
        />
      </Card>
    </Space>
  );
};

export default ProjectList;
