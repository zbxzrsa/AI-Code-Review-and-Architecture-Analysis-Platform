import React, { createContext, useState, useContext, ReactNode, useEffect } from 'react';
import { message } from 'antd';

interface FeedbackContextType {
  showSuccess: (content: string) => void;
  showError: (content: string) => void;
  showWarning: (content: string) => void;
  showInfo: (content: string) => void;
}

const FeedbackContext = createContext<FeedbackContextType | undefined>(undefined);

export const FeedbackProvider: React.FC<{ children: ReactNode }> = ({ children }) => {
  const [messageApi, contextHolder] = message.useMessage();
  useEffect(() => {
    message.config({ duration: 2, maxCount: 3 });
  }, []);

  const showSuccess = (content: string) => {
    messageApi.success(content);
  };

  const showError = (content: string) => {
    messageApi.error(content);
  };

  const showWarning = (content: string) => {
    messageApi.warning(content);
  };

  const showInfo = (content: string) => {
    messageApi.info(content);
  };

  return (
    <FeedbackContext.Provider value={{ showSuccess, showError, showWarning, showInfo }}>
      {contextHolder}
      {children}
    </FeedbackContext.Provider>
  );
};

export const useFeedback = (): FeedbackContextType => {
  const context = useContext(FeedbackContext);
  if (!context) {
    throw new Error('useFeedback must be used within a FeedbackProvider');
  }
  return context;
};
