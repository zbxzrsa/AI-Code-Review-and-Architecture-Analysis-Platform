import React, { useState, FormEvent } from 'react';
import { Form, Input, Button, Alert, message } from 'antd';
import { LockOutlined, SafetyOutlined } from '@ant-design/icons';
import { sanitizeInput, isValidEmail, checkPasswordStrength } from '../../utils/securityUtils';

interface SecureFormProps {
  onSubmit: (values: Record<string, string>) => Promise<void>;
  fields: Array<{
    name: string;
    label: string;
    type: 'text' | 'email' | 'password' | 'textarea';
    required?: boolean;
    placeholder?: string;
    rules?: Array<{
      validator?: (value: string) => boolean;
      message: string;
    }>;
  }>;
  submitText?: string;
  title?: string;
}

/**
 * 安全表单组件
 * 提供输入验证、XSS防护和密码强度检查
 */
const SecureForm: React.FC<SecureFormProps> = ({
  onSubmit,
  fields,
  submitText = 'Submit',
  title,
}) => {
  const [form] = Form.useForm();
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [passwordStrength, setPasswordStrength] = useState(0);

  const handleSubmit = async (values: Record<string, string>) => {
    try {
      setLoading(true);
      setError(null);
      
      // 清理所有输入，防止XSS攻击
      const sanitizedValues = Object.keys(values).reduce((acc, key) => {
        if (typeof values[key] === 'string') {
          acc[key] = sanitizeInput(values[key]);
        } else {
          acc[key] = values[key];
        }
        return acc;
      }, {} as Record<string, any>);
      
      await onSubmit(sanitizedValues);
      form.resetFields();
      message.success('Operation successful');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Submission failed, please try again');
    } finally {
      setLoading(false);
    }
  };

  const validateField = (field: SecureFormProps['fields'][0], value: string) => {
    if (field.required && !value) {
      return Promise.reject(new Error(`${field.label} cannot be empty`));
    }
    
    if (field.type === 'email' && value && !isValidEmail(value)) {
      return Promise.reject(new Error('Please enter a valid email address'));
    }
    
    if (field.rules) {
      for (const rule of field.rules) {
        if (rule.validator && !rule.validator(value)) {
          return Promise.reject(new Error(rule.message));
        }
      }
    }
    
    return Promise.resolve();
  };

  const handlePasswordChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const strength = checkPasswordStrength(e.target.value);
    setPasswordStrength(strength);
  };

  const getPasswordStrengthColor = () => {
    if (passwordStrength < 40) return '#ff4d4f';
    if (passwordStrength < 70) return '#faad14';
    return '#52c41a';
  };

  return (
    <div className="secure-form">
      {title && <h2>{title}</h2>}
      
      {error && (
        <Alert
          message="Error"
          description={error}
          type="error"
          showIcon
          closable
          style={{ marginBottom: 16 }}
        />
      )}
      
      <Form
        form={form}
        layout="vertical"
        onFinish={handleSubmit}
        autoComplete="off"
      >
        {fields.map((field) => (
          <Form.Item
            key={field.name}
            name={field.name}
            label={field.label}
            rules={[
              {
                validator: async (_, value) => validateField(field, value),
              },
            ]}
          >
            {field.type === 'password' ? (
              <>
                <Input.Password
                  prefix={<LockOutlined />}
                  placeholder={field.placeholder}
                  onChange={handlePasswordChange}
                  autoComplete="current-password"
                />
                {passwordStrength > 0 && (
                  <div style={{ marginTop: 8 }}>
                    <div
                      style={{
                        height: 4,
                        background: '#f0f0f0',
                        borderRadius: 2,
                        overflow: 'hidden',
                      }}
                    >
                      <div
                        style={{
                          height: '100%',
                          width: `${passwordStrength}%`,
                          background: getPasswordStrengthColor(),
                          transition: 'all 0.3s',
                        }}
                      />
                    </div>
                     <div style={{ marginTop: 4, fontSize: 12, color: getPasswordStrengthColor() }}>
                       {passwordStrength < 40 && 'Weak Password'}
                       {passwordStrength >= 40 && passwordStrength < 70 && 'Medium Strength'}
                       {passwordStrength >= 70 && 'Strong Password'}
                     </div>
                  </div>
                )}
              </>
            ) : field.type === 'textarea' ? (
              <Input.TextArea placeholder={field.placeholder} />
            ) : field.type === 'email' ? (
              <Input
                type="email"
                placeholder={field.placeholder}
              />
            ) : (
              <Input placeholder={field.placeholder} />
            )}
          </Form.Item>
        ))}
        
        <Form.Item>
          <Button
            type="primary"
            htmlType="submit"
            loading={loading}
            icon={<SafetyOutlined />}
            block
          >
            {submitText}
          </Button>
        </Form.Item>
      </Form>
    </div>
  );
};

export default SecureForm;