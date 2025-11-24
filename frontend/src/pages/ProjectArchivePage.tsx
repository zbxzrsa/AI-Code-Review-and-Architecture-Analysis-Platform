import React from 'react';
import { Card, Typography, List, Button, Space, Result } from 'antd';
import { useNavigate } from 'react-router-dom';
import {
  ProjectRecord,
  projectRepository,
} from '../services/projectRepository';

const { Title, Paragraph } = Typography;

const ProjectArchivePage: React.FC = () => {
  const navigate = useNavigate();
  const archivedProjects = projectRepository
    .list()
    .filter((project) => project.status === 'Archived');

  const handleRestore = (project: ProjectRecord) => {
    projectRepository.update(project.id, { status: 'Active' });
    navigate('/projects');
  };

  if (archivedProjects.length === 0) {
    return (
      <Result
        status="success"
        title="No archived projects"
        subTitle="Great news: everything is active. Use the danger zone to archive a project if needed."
        extra={
          <Button type="primary" onClick={() => navigate('/projects')}>
            Back to projects
          </Button>
        }
      />
    );
  }

  return (
    <Card>
      <Title level={3}>Archived projects</Title>
      <Paragraph type="secondary">
        Restore a project to resume analysis sessions, or delete it permanently via the
        danger zone view.
      </Paragraph>
      <List
        dataSource={archivedProjects}
        renderItem={(project) => (
          <List.Item
            actions={[
              <Button key="restore" type="link" onClick={() => handleRestore(project)}>
                Restore
              </Button>,
              <Button
                key="danger"
                type="link"
                onClick={() => navigate(`/projects/${project.id}/danger-zone`)}
              >
                Danger zone
              </Button>,
            ]}
          >
            <List.Item.Meta
              title={project.name}
              description={
                <Space direction="vertical">
                  <Paragraph style={{ marginBottom: 0 }}>{project.description}</Paragraph>
                  <Paragraph type="secondary" style={{ marginBottom: 0 }}>
                    Last updated: {new Date(project.updatedAt).toLocaleString()}
                  </Paragraph>
                </Space>
              }
            />
          </List.Item>
        )}
      />
    </Card>
  );
};

export default ProjectArchivePage;

