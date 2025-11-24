import React, { useMemo } from 'react';
import {
  Card,
  Typography,
  Descriptions,
  Space,
  Button,
  Tag,
  List,
  Timeline,
  Result,
  Statistic,
  Row,
  Col,
  Divider,
} from 'antd';
import { useNavigate, useParams } from 'react-router-dom';
import {
  ProjectRecord,
  projectRepository,
} from '../services/projectRepository';

const { Title, Paragraph, Text } = Typography;

const ProjectDetail: React.FC = () => {
  const navigate = useNavigate();
  const { id } = useParams<{ id: string }>();
  const project: ProjectRecord | undefined = id
    ? projectRepository.get(id)
    : undefined;

  const activities = useMemo(
    () => projectRepository.listActivities(id),
    [id]
  );

  const health = useMemo(() => {
    if (!project) {
      return { velocity: 0, coverage: 0, openFindings: 0 };
    }
    return {
      velocity: Math.min(100, 60 + project.tags.length * 5),
      coverage: Math.min(100, 70 + project.qualityScore / 3),
      openFindings: Math.max(0, Math.round(project.riskScore / 5)),
    };
  }, [project]);

  if (!project) {
    return (
      <Result
        status="404"
        title="Project not found"
        subTitle="The requested project does not exist. Please use the portfolio page to select a valid project."
        extra={
          <Button type="primary" onClick={() => navigate('/projects')}>
            Back to projects
          </Button>
        }
      />
    );
  }

  return (
    <Space direction="vertical" size="large" style={{ width: '100%' }}>
      <Card
        title={project.name}
        extra={
          <Space>
            <Button onClick={() => navigate(`/projects/${project.id}/edit`)}>
              Edit project
            </Button>
            <Button onClick={() => navigate(`/sessions?projectId=${project.id}`)}>
              View sessions
            </Button>
            <Button
              type="primary"
              onClick={() =>
                navigate(`/analysis?projectId=${encodeURIComponent(project.id)}`)
              }
            >
              Launch analysis
            </Button>
          </Space>
        }
      >
        <Row gutter={16}>
          <Col span={12}>
            <Descriptions bordered column={1} size="small">
              <Descriptions.Item label="Owner">{project.owner}</Descriptions.Item>
              <Descriptions.Item label="Status">{project.status}</Descriptions.Item>
              <Descriptions.Item label="Last analysis">
                {project.lastAnalysis
                  ? new Date(project.lastAnalysis).toLocaleString()
                  : 'Never'}
              </Descriptions.Item>
              <Descriptions.Item label="Repository">
                {project.repoUrl ? (
                  <a href={project.repoUrl} target="_blank" rel="noreferrer">
                    {project.repoUrl}
                  </a>
                ) : (
                  'Not linked'
                )}
              </Descriptions.Item>
              <Descriptions.Item label="Demo workspace">
                {project.demoLink ? (
                  <a href={project.demoLink} target="_blank" rel="noreferrer">
                    {project.demoLink}
                  </a>
                ) : (
                  'Not published'
                )}
              </Descriptions.Item>
            </Descriptions>
          </Col>
          <Col span={12}>
            <Paragraph>{project.description}</Paragraph>
            <Space wrap>
              {project.tags.map((tag) => (
                <Tag key={tag}>{tag}</Tag>
              ))}
            </Space>
            <Divider />
            <Row gutter={16}>
              <Col span={8}>
                <Statistic
                  title="Quality"
                  value={project.qualityScore}
                  suffix="/ 100"
                />
              </Col>
              <Col span={8}>
                <Statistic
                  title="Velocity"
                  value={health.velocity}
                  suffix="%"
                  valueStyle={{ color: '#52c41a' }}
                />
              </Col>
              <Col span={8}>
                <Statistic
                  title="Open findings"
                  value={health.openFindings}
                  valueStyle={{ color: '#faad14' }}
                />
              </Col>
            </Row>
          </Col>
        </Row>
      </Card>

      <Row gutter={16}>
        <Col span={12}>
          <Card title="Activity timeline">
            <Timeline
              items={activities.slice(0, 6).map((activity) => ({
                color:
                  activity.type === 'alert'
                    ? 'red'
                    : activity.type === 'release'
                    ? 'green'
                    : 'blue',
                children: (
                  <div>
                    <Text strong>{activity.title}</Text>
                    <Paragraph style={{ marginBottom: 4 }}>
                      {activity.description}
                    </Paragraph>
                    <Text type="secondary">
                      {new Date(activity.timestamp).toLocaleString()} ·{' '}
                      {activity.actor}
                    </Text>
                  </div>
                ),
              }))}
            />
          </Card>
        </Col>
        <Col span={12}>
          <Card title="Example demonstrations">
            <List
              dataSource={[
                {
                  title: 'Drift remediation sprint',
                  description:
                    'Start from the latest architecture session, capture diff highlights, and assign follow-up tasks.',
                  action: () =>
                    navigate(`/sessions?projectId=${encodeURIComponent(project.id)}`),
                  button: 'Open session tracker',
                },
                {
                  title: 'Baseline promotion',
                  description:
                    'Compare current metrics with baseline and publish the next milestone.',
                  action: () => navigate('/baselines'),
                  button: 'Compare baselines',
                },
                {
                  title: 'Version audit',
                  description:
                    'Open Version Intelligence, load the branch, and export the diff-packed PDF.',
                  action: () => navigate('/versions'),
                  button: 'Inspect versions',
                },
              ]}
              renderItem={(item) => (
                <List.Item
                  actions={[
                    <Button type="link" key={item.title} onClick={item.action}>
                      {item.button}
                    </Button>,
                  ]}
                >
                  <List.Item.Meta title={item.title} description={item.description} />
                </List.Item>
              )}
            />
          </Card>
        </Col>
      </Row>
    </Space>
  );
};

export default ProjectDetail;
