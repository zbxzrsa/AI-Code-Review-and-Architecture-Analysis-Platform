import React from 'react';
import { Navigate } from 'react-router-dom';
import { useAuth } from '../contexts/AuthContext';

interface PrivateRouteProps {
  children: React.ReactNode;
}

const PrivateRoute: React.FC<PrivateRouteProps> = ({ children }) => {
  const { isAuthenticated } = useAuth();
  
  if (!isAuthenticated) {
    // 用户未认证，重定向到登录页面
    return <Navigate to="/login" replace />;
  }

  // 用户已认证，渲染子组件
  return <>{children}</>;
};

export default PrivateRoute;