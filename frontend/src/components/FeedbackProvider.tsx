import React, { createContext, useContext, useState, useCallback } from 'react';
import { message, notification } from 'antd';
import { CheckCircleOutlined, InfoCircleOutlined, WarningOutlined, CloseCircleOutlined } from '@ant-design/icons';

// 反馈上下文类型
interface FeedbackContextType {
  showSuccess: (content: string, duration?: number) => void;
  showInfo: (content: string, duration?: number) => void;
  showWarning: (content: string, duration?: number) => void;
  showError: (content: string, duration?: number) => void;
  showNotification: (type: 'success' | 'info' | 'warning' | 'error', title: string, content: string) => void;
  showLoading: (content: string) => void;
  hideLoading: () => void;
  announceToScreenReader: (message: string, politeness?: 'polite' | 'assertive') => void;
}

// 创建反馈上下文
const FeedbackContext = createContext<FeedbackContextType | undefined>(undefined);

// 反馈提供者组件
export const FeedbackProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [loadingKey, setLoadingKey] = useState<string | null>(null);

  // 成功消息
  const showSuccess = useCallback((content: string, duration = 3) => {
    message.success({
      content,
      duration,
      icon: <CheckCircleOutlined />,
    });
    announceToScreenReader(`成功: ${content}`, 'polite');
  }, []);

  // 信息消息
  const showInfo = useCallback((content: string, duration = 3) => {
    message.info({
      content,
      duration,
      icon: <InfoCircleOutlined />,
    });
    announceToScreenReader(`信息: ${content}`, 'polite');
  }, []);

  // 警告消息
  const showWarning = useCallback((content: string, duration = 4) => {
    message.warning({
      content,
      duration,
      icon: <WarningOutlined />,
    });
    announceToScreenReader(`警告: ${content}`, 'assertive');
  }, []);

  // 错误消息
  const showError = useCallback((content: string, duration = 5) => {
    message.error({
      content,
      duration,
      icon: <CloseCircleOutlined />,
    });
    announceToScreenReader(`错误: ${content}`, 'assertive');
  }, []);

  // 通知
  const showNotification = useCallback((type: 'success' | 'info' | 'warning' | 'error', title: string, content: string) => {
    notification[type]({
      message: title,
      description: content,
    });
    announceToScreenReader(`${title}: ${content}`, type === 'error' || type === 'warning' ? 'assertive' : 'polite');
  }, []);

  // 加载消息
  const showLoading = useCallback((content: string) => {
    const key = Date.now().toString();
    setLoadingKey(key);
    message.loading({
      content,
      duration: 0,
      key,
    });
    announceToScreenReader(`正在加载: ${content}`, 'polite');
  }, []);

  // 隐藏加载消息
  const hideLoading = useCallback(() => {
    if (loadingKey) {
      message.destroy(loadingKey);
      setLoadingKey(null);
    }
  }, [loadingKey]);

  // 屏幕阅读器通知
  const announceToScreenReader = useCallback((message: string, politeness: 'polite' | 'assertive' = 'polite') => {
    let announcer = document.getElementById(`sr-announcer-${politeness}`);
    
    if (!announcer) {
      announcer = document.createElement('div');
      announcer.id = `sr-announcer-${politeness}`;
      announcer.setAttribute('aria-live', politeness);
      announcer.setAttribute('role', 'status');
      announcer.setAttribute('aria-atomic', 'true');
      
      // 设置样式使其对视觉用户不可见，但对屏幕阅读器可访问
      announcer.style.position = 'absolute';
      announcer.style.width = '1px';
      announcer.style.height = '1px';
      announcer.style.padding = '0';
      announcer.style.margin = '-1px';
      announcer.style.overflow = 'hidden';
      announcer.style.clip = 'rect(0, 0, 0, 0)';
      announcer.style.whiteSpace = 'nowrap';
      announcer.style.border = '0';
      
      document.body.appendChild(announcer);
    }
    
    // 清空后重新设置内容，确保屏幕阅读器能够识别内容变化
    announcer.textContent = '';
    
    // 使用setTimeout确保DOM更新
    setTimeout(() => {
      if (announcer) {
        announcer.textContent = message;
      }
    }, 50);
  }, []);

  // 提供上下文值
  const contextValue: FeedbackContextType = {
    showSuccess,
    showInfo,
    showWarning,
    showError,
    showNotification,
    showLoading,
    hideLoading,
    announceToScreenReader,
  };

  return (
    <FeedbackContext.Provider value={contextValue}>
      {children}
    </FeedbackContext.Provider>
  );
};

// 使用反馈钩子
export const useFeedback = (): FeedbackContextType => {
  const context = useContext(FeedbackContext);
  if (!context) {
    throw new Error('useFeedback must be used within a FeedbackProvider');
  }
  return context;
};

export default FeedbackProvider;