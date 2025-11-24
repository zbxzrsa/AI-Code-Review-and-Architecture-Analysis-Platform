import React from 'react';
import {
  Card,
  Typography,
  Steps,
  Form,
  Input,
  Button,
  Space,
  Alert,
  List,
} from 'antd';
import { useNavigate } from 'react-router-dom';

const { Title, Paragraph } = Typography;

const ProjectImportPage: React.FC = () => {
  const [form] = Form.useForm();
  const navigate = useNavigate();

  const handleImport = () => {
    form.validateFields().then(() => {
      window.open('/github-connect', '_self');
    });
  };

  return (
    <Space direction="vertical" size="large" style={{ width: '100%' }}>
      <Card>
        <Title level={3}>Import projects from GitHub</Title>
        <Paragraph>
          Authenticate through GitHub Connect, choose repositories, and watch the platform
          hydrate metadata, branch protections, and analysis sessions automatically.
        </Paragraph>
        <Steps
          items={[
            { title: 'Connect GitHub', description: 'Complete the OAuth handshake' },
            { title: 'Select repos', description: 'Pick branches to track' },
            { title: 'Confirm metadata', description: 'Assign owners and tags' },
          ]}
          current={1}
        />
        <Alert
          message="Tip"
          description="We store only repository metadata. Source code never leaves your GitHub account."
          type="info"
          showIcon
          style={{ margin: '24px 0' }}
        />
        <Form layout="vertical" form={form}>
          <Form.Item
            label="GitHub Organization or User"
            name="owner"
            rules={[{ required: true, message: 'Enter the owner name' }]}
          >
            <Input placeholder="example-org" />
          </Form.Item>
          <Form.Item
            label="Repository filter"
            name="repository"
            extra="Use wildcards, for example: *app or mobile-*"
          >
            <Input placeholder="platform-*" />
          </Form.Item>
          <Form.Item
            label="Default owner"
            name="defaultOwner"
            rules={[{ required: true, message: 'Describe the accountable team' }]}
          >
            <Input placeholder="Core Architecture" />
          </Form.Item>
          <Space>
            <Button onClick={() => navigate('/projects')}>Cancel</Button>
            <Button type="primary" onClick={handleImport}>
              Launch GitHub Connect
            </Button>
          </Space>
        </Form>
      </Card>

      <Card title="Example demonstrations">
        <List
          dataSource={[
            {
              title: 'Mirroring an existing portfolio',
              description:
                'Import repos under the same GitHub org, then map them to the legacy baselines you already track.',
            },
            {
              title: 'Onboarding a new team',
              description:
                'Create a dedicated GitHub account for pilots, import the sandbox repos, and showcase the end-to-end workflow.',
            },
            {
              title: 'Selective imports',
              description:
                'Filter to mobile-* to avoid pulling the entire monorepo while you are experimenting.',
            },
          ]}
          renderItem={(item) => (
            <List.Item>
              <List.Item.Meta title={item.title} description={item.description} />
            </List.Item>
          )}
        />
      </Card>
    </Space>
  );
};

export default ProjectImportPage;

