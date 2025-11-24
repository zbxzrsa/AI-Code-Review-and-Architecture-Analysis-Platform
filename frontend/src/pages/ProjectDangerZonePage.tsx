import React from 'react';
import {
  Card,
  Typography,
  Alert,
  Button,
  Space,
  Result,
  Divider,
  List,
  message,
} from 'antd';
import { useNavigate, useParams } from 'react-router-dom';
import {
  ProjectRecord,
  projectRepository,
} from '../services/projectRepository';

const { Title, Paragraph } = Typography;

const ProjectDangerZonePage: React.FC = () => {
  const navigate = useNavigate();
  const { id } = useParams<{ id: string }>();
  const project: ProjectRecord | undefined = id
    ? projectRepository.get(id)
    : undefined;

  if (!project) {
    return (
      <Result
        status="404"
        title="Project not found"
        subTitle="Select a valid project before entering the danger zone."
        extra={
          <Button type="primary" onClick={() => navigate('/projects')}>
            Back to projects
          </Button>
        }
      />
    );
  }

  const handleArchive = () => {
    projectRepository.update(project.id, { status: 'Archived' });
    message.success('Project archived. You can restore it from the archive view.');
    navigate('/projects');
  };

  const handleReset = () => {
    projectRepository.reset();
    message.success('Sample data restored. All demo projects have been reset.');
    navigate('/projects');
  };

  const handleDelete = () => {
    projectRepository.remove(project.id);
    message.success('Project deleted permanently');
    navigate('/projects');
  };

  return (
    <Space direction="vertical" size="large" style={{ width: '100%' }}>
      <Card>
        <Title level={3}>Danger zone</Title>
        <Paragraph type="secondary">
          These actions are irreversible. Use them when you need to retire or reset a
          project record. All related sessions, baselines, and activities will be detached.
        </Paragraph>
        <Alert
          type="error"
          message="Deleting a project immediately removes all local demo data."
          showIcon
          style={{ marginBottom: 24 }}
        />
        <Space direction="vertical" style={{ width: '100%' }}>
          <Card type="inner" title="Archive project">
            <Paragraph>
              Move the project into an archived state. Analysis history remains available
              but new sessions are paused.
            </Paragraph>
            <Button onClick={handleArchive}>Archive project</Button>
          </Card>
          <Card type="inner" title="Reset sample data">
            <Paragraph>
              Restore the default demonstration data set for quick demos or onboarding
              sessions.
            </Paragraph>
            <Button onClick={handleReset}>Restore defaults</Button>
          </Card>
          <Card type="inner" title="Delete project forever">
            <Paragraph>
              Remove the project plus all local data. Only do this if you are certain the
              project is no longer needed.
            </Paragraph>
            <Button danger onClick={handleDelete}>
              Delete permanently
            </Button>
          </Card>
        </Space>
      </Card>

      <Card title="Example demonstrations">
        <List
          dataSource={[
            'Retire deprecated branches before merging to main',
            'Reset demo data at the end of a workshop',
            'Archive temporary projects created for spike investigations',
          ]}
          renderItem={(item) => (
            <List.Item>
              <Paragraph style={{ marginBottom: 0 }}>{item}</Paragraph>
            </List.Item>
          )}
        />
        <Divider />
        <Paragraph type="secondary">
          Need to rehydrate data from GitHub? Use the import workflow afterwards to
          re-register the repository and continue where you left off.
        </Paragraph>
      </Card>
    </Space>
  );
};

export default ProjectDangerZonePage;

