import React, { useState } from 'react';
import { Card, Form, Input, Button, Avatar, Upload, Space, Typography, Divider, Row, Col, Tag, Switch, message } from 'antd';
import { UserOutlined, CameraOutlined, SaveOutlined, EditOutlined, MailOutlined, PhoneOutlined, GlobalOutlined } from '@ant-design/icons';
import type { UploadProps } from 'antd';

const { Title, Text, Paragraph } = Typography;

interface UserProfile {
  name: string;
  email: string;
  phone: string;
  location: string;
  bio: string;
  company: string;
  role: string;
  website: string;
  github: string;
  linkedin: string;
  notifications: {
    email: boolean;
    push: boolean;
    analysis: boolean;
    security: boolean;
  };
}

const Profile: React.FC = () => {
  const [form] = Form.useForm();
  const [editing, setEditing] = useState(false);
  const [loading, setLoading] = useState(false);
  const [avatarUrl, setAvatarUrl] = useState<string>('');

  // Mock user data
  const [profile, setProfile] = useState<UserProfile>({
    name: 'John Doe',
    email: 'john.doe@example.com',
    phone: '+1 (555) 123-4567',
    location: 'San Francisco, CA',
    bio: 'Senior Software Engineer with 10+ years of experience in full-stack development and code quality analysis.',
    company: 'Tech Corp',
    role: 'Senior Developer',
    website: 'https://johndoe.dev',
    github: 'https://github.com/johndoe',
    linkedin: 'https://linkedin.com/in/johndoe',
    notifications: {
      email: true,
      push: true,
      analysis: true,
      security: true
    }
  });

  const handleEdit = () => {
    setEditing(true);
    form.setFieldsValue(profile);
  };

  const handleCancel = () => {
    setEditing(false);
    form.resetFields();
  };

  const handleSave = async (values: Partial<UserProfile>) => {
    setLoading(true);
    try {
      // Simulate API call
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      setProfile(prev => ({ ...prev, ...values }));
      setEditing(false);
      message.success('Profile updated successfully!');
    } catch (error) {
      message.error('Failed to update profile');
    } finally {
      setLoading(false);
    }
  };

  const handleNotificationChange = (key: keyof UserProfile['notifications'], value: boolean) => {
    setProfile(prev => ({
      ...prev,
      notifications: {
        ...prev.notifications,
        [key]: value
      }
    }));
  };

  const uploadProps: UploadProps = {
    name: 'avatar',
    showUploadList: false,
    beforeUpload: (file) => {
      const isJpgOrPng = file.type === 'image/jpeg' || file.type === 'image/png';
      if (!isJpgOrPng) {
        message.error('You can only upload JPG/PNG file!');
        return false;
      }
      const isLt2M = file.size / 1024 / 1024 < 2;
      if (!isLt2M) {
        message.error('Image must smaller than 2MB!');
        return false;
      }
      
      // Create preview
      const reader = new FileReader();
      reader.onload = (e) => {
        setAvatarUrl(e.target?.result as string);
      };
      reader.readAsDataURL(file);
      
      return false; // Prevent upload
    },
  };

  const skills = ['React', 'TypeScript', 'Python', 'Node.js', 'AWS', 'Docker', 'Kubernetes'];
  const achievements = [
    { name: 'Code Review Master', description: 'Completed 100+ code reviews', icon: '🏆' },
    { name: 'Security Expert', description: 'Identified 50+ security vulnerabilities', icon: '🛡️' },
    { name: 'Performance Guru', description: 'Optimized 20+ applications', icon: '⚡' }
  ];

  return (
    <div style={{ padding: '24px', maxWidth: '1200px', margin: '0 auto' }}>
      <Row gutter={[24, 24]}>
        {/* Profile Information */}
        <Col xs={24} lg={16}>
          <Card
            title={
              <Space>
                <UserOutlined />
                Profile Information
                {!editing && (
                  <Button size="small" icon={<EditOutlined />} onClick={handleEdit}>
                    Edit
                  </Button>
                )}
              </Space>
            }
            extra={
              editing && (
                <Space>
                  <Button onClick={handleCancel}>Cancel</Button>
                  <Button 
                    type="primary" 
                    icon={<SaveOutlined />}
                    loading={loading}
                    onClick={() => form.submit()}
                  >
                    Save
                  </Button>
                </Space>
              )
            }
          >
            {editing ? (
              <Form
                form={form}
                layout="vertical"
                onFinish={handleSave}
              >
                <Row gutter={16}>
                  <Col xs={24} md={12}>
                    <Form.Item label="Full Name" name="name" rules={[{ required: true }]}>
                      <Input prefix={<UserOutlined />} />
                    </Form.Item>
                  </Col>
                  <Col xs={24} md={12}>
                    <Form.Item label="Email" name="email" rules={[{ required: true, type: 'email' }]}>
                      <Input prefix={<MailOutlined />} />
                    </Form.Item>
                  </Col>
                </Row>
                
                <Row gutter={16}>
                  <Col xs={24} md={12}>
                    <Form.Item label="Phone" name="phone">
                      <Input prefix={<PhoneOutlined />} />
                    </Form.Item>
                  </Col>
                  <Col xs={24} md={12}>
                    <Form.Item label="Location" name="location">
                      <Input prefix={<GlobalOutlined />} />
                    </Form.Item>
                  </Col>
                </Row>

                <Row gutter={16}>
                  <Col xs={24} md={12}>
                    <Form.Item label="Company" name="company">
                      <Input />
                    </Form.Item>
                  </Col>
                  <Col xs={24} md={12}>
                    <Form.Item label="Role" name="role">
                      <Input />
                    </Form.Item>
                  </Col>
                </Row>

                <Form.Item label="Bio" name="bio">
                  <Input.TextArea rows={3} />
                </Form.Item>

                <Row gutter={16}>
                  <Col xs={24} md={8}>
                    <Form.Item label="Website" name="website">
                      <Input placeholder="https://..." />
                    </Form.Item>
                  </Col>
                  <Col xs={24} md={8}>
                    <Form.Item label="GitHub" name="github">
                      <Input placeholder="https://github.com/..." />
                    </Form.Item>
                  </Col>
                  <Col xs={24} md={8}>
                    <Form.Item label="LinkedIn" name="linkedin">
                      <Input placeholder="https://linkedin.com/in/..." />
                    </Form.Item>
                  </Col>
                </Row>
              </Form>
            ) : (
              <div>
                <Row gutter={[24, 16]}>
                  <Col xs={24} md={8}>
                    <Text strong>Full Name</Text>
                    <br />
                    <Text>{profile.name}</Text>
                  </Col>
                  <Col xs={24} md={8}>
                    <Text strong>Email</Text>
                    <br />
                    <Text>{profile.email}</Text>
                  </Col>
                  <Col xs={24} md={8}>
                    <Text strong>Phone</Text>
                    <br />
                    <Text>{profile.phone}</Text>
                  </Col>
                </Row>

                <Divider />

                <Row gutter={[24, 16]}>
                  <Col xs={24} md={8}>
                    <Text strong>Location</Text>
                    <br />
                    <Text>{profile.location}</Text>
                  </Col>
                  <Col xs={24} md={8}>
                    <Text strong>Company</Text>
                    <br />
                    <Text>{profile.company}</Text>
                  </Col>
                  <Col xs={24} md={8}>
                    <Text strong>Role</Text>
                    <br />
                    <Text>{profile.role}</Text>
                  </Col>
                </Row>

                <Divider />

                <div style={{ marginBottom: 16 }}>
                  <Text strong>Bio</Text>
                  <br />
                  <Paragraph>{profile.bio}</Paragraph>
                </div>

                <Row gutter={16}>
                  <Col xs={24} md={8}>
                    <Text strong>Website</Text>
                    <br />
                    <a href={profile.website} target="_blank" rel="noopener noreferrer">
                      {profile.website}
                    </a>
                  </Col>
                  <Col xs={24} md={8}>
                    <Text strong>GitHub</Text>
                    <br />
                    <a href={profile.github} target="_blank" rel="noopener noreferrer">
                      {profile.github}
                    </a>
                  </Col>
                  <Col xs={24} md={8}>
                    <Text strong>LinkedIn</Text>
                    <br />
                    <a href={profile.linkedin} target="_blank" rel="noopener noreferrer">
                      {profile.linkedin}
                    </a>
                  </Col>
                </Row>
              </div>
            )}
          </Card>

          {/* Skills */}
          <Card title="Skills & Expertise" style={{ marginTop: 24 }}>
            <Space wrap>
              {skills.map(skill => (
                <Tag key={skill} color="blue">{skill}</Tag>
              ))}
            </Space>
          </Card>
        </Col>

        {/* Sidebar */}
        <Col xs={24} lg={8}>
          {/* Avatar */}
          <Card title="Profile Picture">
            <div style={{ textAlign: 'center' }}>
              <Avatar
                size={120}
                src={avatarUrl}
                icon={<UserOutlined />}
                style={{ marginBottom: 16 }}
              />
              <br />
              <Upload {...uploadProps}>
                <Button icon={<CameraOutlined />} size="small">
                  Change Avatar
                </Button>
              </Upload>
            </div>
          </Card>

          {/* Achievements */}
          <Card title="Achievements" style={{ marginTop: 24 }}>
            <Space direction="vertical" style={{ width: '100%' }}>
              {achievements.map((achievement, index) => (
                <div key={index} style={{ 
                  padding: '12px', 
                  border: '1px solid #f0f0f0', 
                  borderRadius: '6px',
                  backgroundColor: '#fafafa'
                }}>
                  <Space>
                    <span style={{ fontSize: '24px' }}>{achievement.icon}</span>
                    <div>
                      <div style={{ fontWeight: 600 }}>{achievement.name}</div>
                      <Text type="secondary" style={{ fontSize: '12px' }}>
                        {achievement.description}
                      </Text>
                    </div>
                  </Space>
                </div>
              ))}
            </Space>
          </Card>

          {/* Notification Settings */}
          <Card title="Notification Settings" style={{ marginTop: 24 }}>
            <Space direction="vertical" style={{ width: '100%' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <Text>Email Notifications</Text>
                <Switch 
                  checked={profile.notifications.email}
                  onChange={(checked) => handleNotificationChange('email', checked)}
                />
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <Text>Push Notifications</Text>
                <Switch 
                  checked={profile.notifications.push}
                  onChange={(checked) => handleNotificationChange('push', checked)}
                />
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <Text>Analysis Updates</Text>
                <Switch 
                  checked={profile.notifications.analysis}
                  onChange={(checked) => handleNotificationChange('analysis', checked)}
                />
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <Text>Security Alerts</Text>
                <Switch 
                  checked={profile.notifications.security}
                  onChange={(checked) => handleNotificationChange('security', checked)}
                />
              </div>
            </Space>
          </Card>
        </Col>
      </Row>
    </div>
  );
};

export default Profile;