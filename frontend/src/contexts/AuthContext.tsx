import React, { createContext, useState, useContext, useEffect, ReactNode } from 'react';

interface User {
  id: string;
  username: string;
  email: string;
}

interface AuthContextType {
  user: User | null;
  isAuthenticated: boolean;
  login: (email: string, password: string) => Promise<boolean>;
  register: (username: string, email: string, password: string) => Promise<boolean>;
  logout: () => void;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export const AuthProvider: React.FC<{ children: ReactNode }> = ({ children }) => {
  const [user, setUser] = useState<User | null>(null);
  const [isAuthenticated, setIsAuthenticated] = useState<boolean>(false);
  const shouldAutoSignIn = process.env.REACT_APP_REQUIRE_LOGIN !== 'true';

  // 初始化时检查本地存储中的用户信息
  useEffect(() => {
    const storedUser = localStorage.getItem('user');
    if (storedUser) {
      try {
        const parsedUser = JSON.parse(storedUser);
        setUser(parsedUser);
        setIsAuthenticated(true);
      } catch (error) {
        console.error('Failed to parse stored user:', error);
        localStorage.removeItem('user');
      }
    } else if (shouldAutoSignIn) {
      const demoUser = {
        id: 'demo',
        username: 'demo_user',
        email: 'demo@example.com'
      };
      setUser(demoUser);
      setIsAuthenticated(true);
      localStorage.setItem('user', JSON.stringify(demoUser));
      localStorage.setItem('auth_token', 'demo-token');
    }
  }, [shouldAutoSignIn]);

  // 登录函数
  const login = async (email: string, password: string): Promise<boolean> => {
    try {
      // 在实际应用中，这里应该调用API进行身份验证
      // 这里简化为模拟成功登录
      const mockUser = {
        id: '1',
        username: email.split('@')[0],
        email
      };
      
      setUser(mockUser);
      setIsAuthenticated(true);
      localStorage.setItem('user', JSON.stringify(mockUser));
      localStorage.setItem('auth_token', 'demo-token');
      return true;
    } catch (error) {
      console.error('Login failed:', error);
      return false;
    }
  };

  // 注册函数
  const register = async (username: string, email: string, password: string): Promise<boolean> => {
    try {
      // 在实际应用中，这里应该调用API进行注册
      // 这里简化为模拟成功注册
      const mockUser = {
        id: Date.now().toString(),
        username,
        email
      };
      
      setUser(mockUser);
      setIsAuthenticated(true);
      localStorage.setItem('user', JSON.stringify(mockUser));
      localStorage.setItem('auth_token', 'demo-token');
      return true;
    } catch (error) {
      console.error('Registration failed:', error);
      return false;
    }
  };

  // 登出函数
  const logout = () => {
    setUser(null);
    setIsAuthenticated(false);
    localStorage.removeItem('user');
    localStorage.removeItem('auth_token');
  };

  return (
    <AuthContext.Provider value={{ user, isAuthenticated, login, register, logout }}>
      {children}
    </AuthContext.Provider>
  );
};

// 自定义钩子，用于在组件中使用认证功能
export const useAuth = (): AuthContextType => {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};