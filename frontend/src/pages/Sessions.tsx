import React, { useState, useEffect } from 'react';
import { Card, Table, Button, Tag, Space, Tooltip, Modal, Form, Input, Select, message, List } from 'antd';
import { PlayCircleOutlined, PauseCircleOutlined, DeleteOutlined, EyeOutlined, PlusOutlined } from '@ant-design/icons';
import type { ColumnsType } from 'antd/es/table';
import { useNavigate } from 'react-router-dom';

interface SessionArtifact {
  id: number;
  type: string;
  path: string;
  size: number;
  created_at: string;
}

interface AnalysisSession {
  id: number;
  project_id: number;
  label: string;
  status: string;
  started_at: string;
  completed_at?: string;
  summary?: string;
  artifacts: SessionArtifact[];
}

const Sessions: React.FC = () => {
  const navigate = useNavigate();
  const [sessions, setSessions] = useState<AnalysisSession[]>([]);
  const [loading, setLoading] = useState(false);
  const [createModalVisible, setCreateModalVisible] = useState(false);
  const [form] = Form.useForm();

  const statusColors: Record<string, string> = {
    'pending': 'orange',
    'running': 'blue',
    'completed': 'green',
    'failed': 'red',
    'cancelled': 'gray'
  };

  const columns: ColumnsType<AnalysisSession> = [
    {
      title: 'ID',
      dataIndex: 'id',
      key: 'id',
      width: 80,
    },
    {
      title: 'Session Label',
      dataIndex: 'label',
      key: 'label',
    },
    {
      title: 'Project ID',
      dataIndex: 'project_id',
      key: 'project_id',
      width: 100,
    },
    {
      title: 'Status',
      dataIndex: 'status',
      key: 'status',
      width: 100,
      render: (status: string) => (
        <Tag color={statusColors[status] || 'default'}>
          {status.toUpperCase()}
        </Tag>
      ),
    },
    {
      title: 'Start Time',
      dataIndex: 'started_at',
      key: 'started_at',
      width: 180,
      render: (date: string) => new Date(date).toLocaleString(),
    },
    {
      title: 'End Time',
      dataIndex: 'completed_at',
      key: 'completed_at',
      width: 180,
      render: (date?: string) => date ? new Date(date).toLocaleString() : '-',
    },
    {
      title: 'Artifacts',
      key: 'artifacts_count',
      width: 100,
      render: (_, record) => record.artifacts.length,
    },
    {
      title: 'Summary',
      dataIndex: 'summary',
      key: 'summary',
      ellipsis: true,
      render: (summary?: string) => summary || '-',
    },
    {
      title: 'Actions',
      key: 'actions',
      width: 200,
      render: (_, record) => (
        <Space>
          <Tooltip title="View Details">
            <Button 
              type="link" 
              icon={<EyeOutlined />} 
              onClick={() => navigateToSession(record.id)}
            />
          </Tooltip>
          {record.status === 'running' && (
            <Tooltip title="Pause Session">
              <Button 
                type="link" 
                icon={<PauseCircleOutlined />} 
                onClick={() => navigateToSession(record.id, 'pause')}
              />
            </Tooltip>
          )}
          {record.status === 'pending' && (
            <Tooltip title="Start Session">
              <Button 
                type="link" 
                icon={<PlayCircleOutlined />} 
                onClick={() => navigateToSession(record.id, 'start')}
              />
            </Tooltip>
          )}
          <Tooltip title="Delete Session">
            <Button 
              type="link" 
              danger 
              icon={<DeleteOutlined />} 
              onClick={() => handleDeleteSession(record.id)}
            />
          </Tooltip>
        </Space>
      ),
    },
  ];

  const fetchSessions = async () => {
    setLoading(true);
    try {
      // TODO: Replace with actual API call
      const mockData: AnalysisSession[] = [
        {
          id: 1,
          project_id: 1,
          label: "Initial Analysis",
          status: "completed",
          started_at: "2024-01-15T10:00:00Z",
          completed_at: "2024-01-15T10:30:00Z",
          summary: "Found 12 issues, 3 security vulnerabilities",
          artifacts: [
            { id: 1, type: "report", path: "/reports/session_1.pdf", size: 2048, created_at: "2024-01-15T10:30:00Z" }
          ]
        },
        {
          id: 2,
          project_id: 1,
          label: "Security Scan",
          status: "running",
          started_at: "2024-01-20T14:00:00Z",
          summary: undefined,
          artifacts: []
        }
      ];
      setSessions(mockData);
    } catch (error) {
      message.error('Failed to fetch sessions');
    } finally {
      setLoading(false);
    }
  };

  const handleCreateSession = async (values: any) => {
    try {
      const newSession: AnalysisSession = {
        id: Date.now(),
        project_id: Number(values.project_id),
        label: values.label,
        status: 'pending',
        started_at: new Date().toISOString(),
        artifacts: [],
      };
      setSessions((prev) => [newSession, ...prev]);
      message.success('Session created successfully');
      setCreateModalVisible(false);
      form.resetFields();
    } catch (error) {
      message.error('Failed to create session');
    }
  };

  const navigateToSession = (sessionId: number, action?: string) => {
    const session = sessions.find((item) => item.id === sessionId);
    const url = action
      ? `/sessions/${sessionId}?action=${action}`
      : `/sessions/${sessionId}`;
    navigate(url, { state: { session, action } });
  };

  const handleDeleteSession = (sessionId: number) => {
    Modal.confirm({
      title: 'Confirm Delete',
      content: 'Are you sure to delete this analysis session? This action cannot be undone.',
      onOk: async () => {
        try {
          setSessions((prev) => prev.filter((session) => session.id !== sessionId));
          message.success('Session deleted successfully');
        } catch (error) {
          message.error('Failed to delete session');
        }
      },
    });
  };

  useEffect(() => {
    fetchSessions();
  }, []);

  return (
    <div style={{ padding: '24px' }}>
      <Card 
        title="Analysis Session Management" 
        extra={
          <Button 
            type="primary" 
            icon={<PlusOutlined />}
            onClick={() => setCreateModalVisible(true)}
          >
            Create Session
          </Button>
        }
      >
        <Table
          columns={columns}
          dataSource={sessions}
          rowKey="id"
          loading={loading}
          pagination={{
            showSizeChanger: true,
            showQuickJumper: true,
            showTotal: (total) => `Total ${total} records`,
          }}
        />
      </Card>

      <Modal
        title="Create Analysis Session"
        open={createModalVisible}
        onCancel={() => setCreateModalVisible(false)}
        onOk={() => form.submit()}
      >
        <Form
          form={form}
          layout="vertical"
          onFinish={handleCreateSession}
        >
          <Form.Item
            name="project_id"
            label="Project ID"
            rules={[{ required: true, message: 'Please enter Project ID' }]}
          >
            <Input type="number" placeholder="Please enter Project ID" />
          </Form.Item>
          <Form.Item
            name="label"
            label="Session Label"
            rules={[{ required: true, message: 'Please enter session label' }]}
          >
            <Input placeholder="Please enter session label" />
          </Form.Item>
        </Form>
      </Modal>

      <Card title="Example demonstrations" style={{ marginTop: 16 }}>
        <List
          dataSource={[
            'Kick off an analysis session after each pull request merge.',
            'Pause a long-running session while investigating unstable environments.',
            'Review artifacts and attach them to a Jira ticket straight from the detail page.',
          ]}
          renderItem={(item) => (
            <List.Item>
              <span>{item}</span>
            </List.Item>
          )}
        />
      </Card>
    </div>
  );
};

export default Sessions;