import React, { useEffect } from 'react';
import {
  Card,
  Typography,
  Form,
  Input,
  Select,
  Button,
  Space,
  message,
  Row,
  Col,
  Slider,
  Divider,
  List,
} from 'antd';
import { useNavigate, useParams } from 'react-router-dom';
import {
  ProjectRecord,
  projectRepository,
} from '../services/projectRepository';

const { Title, Paragraph, Text } = Typography;

const statusOptions: ProjectRecord['status'][] = ['Active', 'On Hold', 'Archived'];

const ProjectFormPage: React.FC = () => {
  const [form] = Form.useForm();
  const navigate = useNavigate();
  const { id } = useParams<{ id: string }>();
  const existingProject = id ? projectRepository.get(id) : undefined;
  const mode = existingProject ? 'edit' : 'create';

  useEffect(() => {
    if (existingProject) {
      form.setFieldsValue({
        name: existingProject.name,
        description: existingProject.description,
        status: existingProject.status,
        owner: existingProject.owner,
        repoUrl: existingProject.repoUrl,
        tags: existingProject.tags,
        qualityScore: existingProject.qualityScore,
        riskScore: existingProject.riskScore,
      });
    } else {
      form.setFieldsValue({
        status: 'Active',
        qualityScore: 75,
        riskScore: 20,
        tags: ['architecture'],
      });
    }
  }, [existingProject, form]);

  const handleSubmit = (values: any) => {
    if (mode === 'create') {
      const created = projectRepository.create({
        name: values.name,
        description: values.description,
        status: values.status,
        owner: values.owner,
        repoUrl: values.repoUrl,
        tags: values.tags || [],
        lastAnalysis: new Date().toISOString(),
        qualityScore: values.qualityScore,
        riskScore: values.riskScore,
        demoLink: values.demoLink,
      });
      message.success('Project created successfully');
      navigate(`/projects/${created.id}`);
    } else if (existingProject) {
      projectRepository.update(existingProject.id, {
        ...existingProject,
        ...values,
      });
      message.success('Project updated successfully');
      navigate(`/projects/${existingProject.id}`);
    }
  };

  return (
    <Space direction="vertical" size="large" style={{ width: '100%' }}>
      <Card>
        <Title level={3}>
          {mode === 'create' ? 'Create a new project' : 'Update project'}
        </Title>
        <Paragraph type="secondary">
          Define ownership, code locations, and guardrails. Every field feeds downstream
          modules such as Baselines, Sessions, and GitHub Connect, so keeping this
          information accurate ensures that automated workflows stay reliable.
        </Paragraph>
        <Form
          layout="vertical"
          form={form}
          onFinish={handleSubmit}
          requiredMark="optional"
        >
          <Row gutter={16}>
            <Col span={12}>
              <Form.Item
                label="Project name"
                name="name"
                rules={[{ required: true, message: 'Please enter a project name' }]}
              >
                <Input placeholder="Graph Orchestration Platform" />
              </Form.Item>
            </Col>
            <Col span={12}>
              <Form.Item
                label="Status"
                name="status"
                rules={[{ required: true, message: 'Please select a status' }]}
              >
                <Select
                  options={statusOptions.map((status) => ({ label: status, value: status }))}
                />
              </Form.Item>
            </Col>
          </Row>
          <Row gutter={16}>
            <Col span={12}>
              <Form.Item
                label="Owner"
                name="owner"
                rules={[{ required: true, message: 'Please define the accountable owner' }]}
              >
                <Input placeholder="Alex Rivera" />
              </Form.Item>
            </Col>
            <Col span={12}>
              <Form.Item label="Repository URL" name="repoUrl">
                <Input placeholder="https://github.com/org/repo" />
              </Form.Item>
            </Col>
          </Row>

          <Form.Item
            label="Description"
            name="description"
            rules={[{ required: true, message: 'Provide a short description' }]}
          >
            <Input.TextArea rows={4} placeholder="Why does this project exist?" />
          </Form.Item>

          <Row gutter={16}>
            <Col span={12}>
              <Form.Item label="Tags" name="tags">
                <Select mode="tags" placeholder="architecture, security, web" />
              </Form.Item>
            </Col>
            <Col span={12}>
              <Form.Item label="Demo link" name="demoLink">
                <Input placeholder="https://demo.company.com/project" />
              </Form.Item>
            </Col>
          </Row>

          <Row gutter={16}>
            <Col span={12}>
              <Form.Item label="Quality score" name="qualityScore">
                <Slider min={30} max={100} tooltip={{ placement: 'bottom' }} />
              </Form.Item>
            </Col>
            <Col span={12}>
              <Form.Item label="Risk score" name="riskScore">
                <Slider min={0} max={60} tooltip={{ placement: 'bottom' }} />
              </Form.Item>
            </Col>
          </Row>

          <Space>
            <Button onClick={() => navigate('/projects')}>Cancel</Button>
            <Button type="primary" htmlType="submit">
              {mode === 'create' ? 'Create project' : 'Save changes'}
            </Button>
          </Space>
        </Form>
      </Card>

      <Card title="Example demonstrations">
        <Row gutter={16}>
          <Col span={8}>
            <Title level={5}>Baseline-ready intake</Title>
            <Paragraph>
              Provide minimum metadata: name, repo link, owner, compliance tags. Submit to
              automatically seed default baselines.
            </Paragraph>
          </Col>
          <Col span={8}>
            <Title level={5}>GitHub-backed projects</Title>
            <Paragraph>
              Import repository, map branches to analysis pipelines, and auto-create release
              sessions using GitHub Connect.
            </Paragraph>
          </Col>
          <Col span={8}>
            <Title level={5}>Architecture rewrites</Title>
            <Paragraph>
              Clone an existing project, change the owner, and switch status to “On Hold” to
              track rewrite trajectories separately.
            </Paragraph>
          </Col>
        </Row>
        <Divider />
        <List
          grid={{ gutter: 16, column: 3 }}
          dataSource={[
            'Run Kickoff wizard',
            'Publish baseline',
            'Add GitHub automation',
          ]}
          renderItem={(item) => (
            <List.Item>
              <Card size="small" title={item}>
                <Text type="secondary">
                  Use the form above, then share the autogenerated link with collaborators.
                </Text>
              </Card>
            </List.Item>
          )}
        />
      </Card>
    </Space>
  );
};

export default ProjectFormPage;

