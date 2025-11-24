import React from 'react';
import { Card, Descriptions, Space, Button, Timeline, Tag, Alert, List } from 'antd';
import { useLocation, useNavigate, useParams, useSearchParams } from 'react-router-dom';

interface SessionRecord {
  id: number | string;
  project_id: string | number;
  label: string;
  status: string;
  started_at?: string;
  completed_at?: string;
  summary?: string;
  artifacts: Array<{ id: number; type: string; path: string }>;
}

const buildFallbackSession = (sessionId: string): SessionRecord => ({
  id: sessionId,
  project_id: 'proj-001',
  label: `Session ${sessionId}`,
  status: 'pending',
  started_at: new Date().toISOString(),
  artifacts: [],
});

const SessionDetail: React.FC = () => {
  const navigate = useNavigate();
  const { state } = useLocation();
  const { id } = useParams<{ id: string }>();
  const [searchParams] = useSearchParams();
  const requestedAction = searchParams.get('action');
  const session: SessionRecord =
    state?.session ?? buildFallbackSession(id || 'unknown');

  return (
    <Space direction="vertical" size="large" style={{ width: '100%' }}>
      <Card
        title={session.label}
        extra={
          <Space>
            <Button onClick={() => navigate('/sessions')}>Back to sessions</Button>
            <Button type="primary" onClick={() => navigate('/analysis')}>
              Launch analysis
            </Button>
          </Space>
        }
      >
        <Descriptions bordered column={2} size="small">
          <Descriptions.Item label="Project">{session.project_id}</Descriptions.Item>
          <Descriptions.Item label="Status">
            <Tag color={session.status === 'completed' ? 'green' : 'orange'}>
              {session.status?.toUpperCase()}
            </Tag>
          </Descriptions.Item>
          <Descriptions.Item label="Started at">
            {session.started_at
              ? new Date(session.started_at).toLocaleString()
              : 'Not started'}
          </Descriptions.Item>
          <Descriptions.Item label="Completed at">
            {session.completed_at
              ? new Date(session.completed_at).toLocaleString()
              : 'In progress'}
          </Descriptions.Item>
        </Descriptions>
        {requestedAction && (
          <Alert
            style={{ marginTop: 16 }}
            message={`Action requested: ${requestedAction}`}
            description="Use the controls below to execute the requested action."
            type="info"
            showIcon
          />
        )}
        <Space style={{ marginTop: 16 }}>
          <Button onClick={() => navigate(`/sessions/${session.id}?action=start`)}>
            Start session
          </Button>
          <Button onClick={() => navigate(`/sessions/${session.id}?action=pause`)}>
            Pause session
          </Button>
          <Button danger onClick={() => navigate(`/sessions/${session.id}?action=cancel`)}>
            Cancel session
          </Button>
        </Space>
      </Card>

      <Card title="Timeline">
        <Timeline
          items={[
            {
              color: 'blue',
              children: 'Session created',
            },
            {
              color: 'green',
              children: 'Static analysis completed',
            },
            {
              color: 'red',
              children: 'Dynamic test flagged a regression',
            },
          ]}
        />
      </Card>

      <Card title="Example demonstrations">
        <List
          dataSource={[
            {
              title: 'Track release readiness',
              description:
                'Open the session after each push to verify that quality gates still pass.',
            },
            {
              title: 'Investigate instability',
              description:
                'Pause the session, collect data, and then resume once the issue is resolved.',
            },
            {
              title: 'Share with stakeholders',
              description:
                'Link the session detail page in your engineering report as a single source of truth.',
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

export default SessionDetail;

