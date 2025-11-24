import React, { useState } from 'react';
import { Drawer, List, Typography, Button, Space, Tag, Empty, Badge, Avatar, Divider } from 'antd';
import { 
  BellOutlined, 
  CheckCircleOutlined, 
  InfoCircleOutlined, 
  WarningOutlined,
  CloseCircleOutlined,
  DeleteOutlined,
  CheckOutlined
} from '@ant-design/icons';

const { Title, Text } = Typography;

interface Notification {
  id: string;
  type: 'info' | 'success' | 'warning' | 'error';
  title: string;
  message: string;
  timestamp: string;
  read: boolean;
  action?: {
    label: string;
    onClick: () => void;
  };
}

interface NotificationsPanelProps {
  visible: boolean;
  onClose: () => void;
  count: number;
  onCountChange: (count: number) => void;
}

const NotificationsPanel: React.FC<NotificationsPanelProps> = ({
  visible,
  onClose,
  count,
  onCountChange
}) => {
  const [notifications, setNotifications] = useState<Notification[]>([
    {
      id: '1',
      type: 'success',
      title: 'Analysis Complete',
      message: 'Code analysis for project "frontend-refactor" has completed successfully. 12 issues found.',
      timestamp: '2 minutes ago',
      read: false,
      action: {
        label: 'View Results',
        onClick: () => console.log('Navigate to analysis results')
      }
    },
    {
      id: '2',
      type: 'warning',
      title: 'High CPU Usage',
      message: 'System CPU usage has exceeded 80% for the last 5 minutes. Consider scaling resources.',
      timestamp: '15 minutes ago',
      read: false,
      action: {
        label: 'View Monitoring',
        onClick: () => console.log('Navigate to monitoring')
      }
    },
    {
      id: '3',
      type: 'info',
      title: 'New Feature Available',
      message: 'AI-powered code suggestions are now available in the analysis panel. Try it out!',
      timestamp: '1 hour ago',
      read: true,
      action: {
        label: 'Learn More',
        onClick: () => console.log('Navigate to help')
      }
    },
    {
      id: '4',
      type: 'error',
      title: 'Analysis Failed',
      message: 'Analysis for project "backend-api" failed due to timeout. Please try again.',
      timestamp: '2 hours ago',
      read: true,
      action: {
        label: 'Retry Analysis',
        onClick: () => console.log('Retry analysis')
      }
    },
    {
      id: '5',
      type: 'success',
      title: 'Project Imported',
      message: 'Successfully imported project "legacy-system" from GitHub repository.',
      timestamp: '3 hours ago',
      read: true
    }
  ]);

  const getIcon = (type: string) => {
    switch (type) {
      case 'success': return <CheckCircleOutlined style={{ color: '#52c41a' }} />;
      case 'warning': return <WarningOutlined style={{ color: '#faad14' }} />;
      case 'error': return <CloseCircleOutlined style={{ color: '#ff4d4f' }} />;
      default: return <InfoCircleOutlined style={{ color: '#1890ff' }} />;
    }
  };

  const getTagColor = (type: string) => {
    switch (type) {
      case 'success': return 'green';
      case 'warning': return 'orange';
      case 'error': return 'red';
      default: return 'blue';
    }
  };

  const markAsRead = (id: string) => {
    setNotifications(prev => 
      prev.map(n => n.id === id ? { ...n, read: true } : n)
    );
    updateCount();
  };

  const markAllAsRead = () => {
    setNotifications(prev => 
      prev.map(n => ({ ...n, read: true }))
    );
    updateCount();
  };

  const deleteNotification = (id: string) => {
    setNotifications(prev => prev.filter(n => n.id !== id));
    updateCount();
  };

  const clearAll = () => {
    setNotifications([]);
    onCountChange(0);
  };

  const updateCount = () => {
    const unreadCount = notifications.filter(n => !n.read).length;
    onCountChange(unreadCount);
  };

  const handleNotificationClick = (notification: Notification) => {
    if (!notification.read) {
      markAsRead(notification.id);
    }
    if (notification.action) {
      notification.action.onClick();
    }
  };

  const unreadCount = notifications.filter(n => !n.read).length;

  return (
    <Drawer
      title={
        <Space>
          <BellOutlined />
          <span>Notifications</span>
          {unreadCount > 0 && (
            <Badge count={unreadCount} size="small" />
          )}
        </Space>
      }
      placement="right"
      width={400}
      open={visible}
      onClose={onClose}
      extra={
        <Space>
          {unreadCount > 0 && (
            <Button 
              size="small" 
              type="text" 
              onClick={markAllAsRead}
              icon={<CheckOutlined />}
            >
              Mark all read
            </Button>
          )}
          {notifications.length > 0 && (
            <Button 
              size="small" 
              type="text" 
              onClick={clearAll}
              icon={<DeleteOutlined />}
            >
              Clear all
            </Button>
          )}
        </Space>
      }
    >
      {notifications.length === 0 ? (
        <Empty
          description="No notifications"
          image={Empty.PRESENTED_IMAGE_SIMPLE}
        />
      ) : (
        <List
          dataSource={notifications}
          renderItem={(notification) => (
            <List.Item
              style={{
                padding: '12px 0',
                borderBottom: '1px solid #f0f0f0',
                cursor: 'pointer',
                backgroundColor: notification.read ? 'transparent' : '#f6ffed',
                borderRadius: '6px',
                paddingLeft: '8px'
              }}
              onClick={() => handleNotificationClick(notification)}
              actions={[
                !notification.read && (
                  <Button
                    size="small"
                    type="text"
                    icon={<CheckOutlined />}
                    onClick={(e) => {
                      e.stopPropagation();
                      markAsRead(notification.id);
                    }}
                  />
                ),
                <Button
                  size="small"
                  type="text"
                  icon={<DeleteOutlined />}
                  onClick={(e) => {
                    e.stopPropagation();
                    deleteNotification(notification.id);
                  }}
                />
              ].filter(Boolean)}
            >
              <List.Item.Meta
                avatar={
                  <Avatar 
                    icon={getIcon(notification.type)} 
                    style={{ 
                      backgroundColor: 'transparent',
                      border: 'none'
                    }} 
                  />
                }
                title={
                  <Space>
                    <Text strong={!notification.read}>
                      {notification.title}
                    </Text>
                    <Tag color={getTagColor(notification.type)}>
                      {notification.type}
                    </Tag>
                  </Space>
                }
                description={
                  <div>
                    <Text type="secondary" style={{ fontSize: '12px' }}>
                      {notification.message}
                    </Text>
                    <br />
                    <Text type="secondary" style={{ fontSize: '11px' }}>
                      {notification.timestamp}
                    </Text>
                    {notification.action && (
                      <div style={{ marginTop: '8px' }}>
                        <Button 
                          size="small" 
                          type="primary"
                          onClick={(e) => {
                            e.stopPropagation();
                            handleNotificationClick(notification);
                          }}
                        >
                          {notification.action.label}
                        </Button>
                      </div>
                    )}
                  </div>
                }
              />
            </List.Item>
          )}
        />
      )}
      
      {notifications.length > 0 && (
        <>
          <Divider />
          <div style={{ textAlign: 'center', padding: '8px 0' }}>
            <Button type="text" size="small">
              View all notifications
            </Button>
          </div>
        </>
      )}
    </Drawer>
  );
};

export default NotificationsPanel;