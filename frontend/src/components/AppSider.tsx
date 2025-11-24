import React, { useMemo, useState } from 'react';
import { Layout, Menu } from 'antd';
import { useNavigate, useLocation } from 'react-router-dom';
import { menuRouteConfigs } from '../routes/appRoutes';

const { Sider } = Layout;

const AppSider: React.FC = () => {
  const [collapsed, setCollapsed] = useState(false);
  const navigate = useNavigate();
  const location = useLocation();
  const items = useMemo(
    () =>
      menuRouteConfigs.map((route) => ({
        key: route.path,
        icon: route.icon,
        label: route.fallbackLabel || route.path,
      })),
    []
  );

  const selectedKey = useMemo(() => {
    if (location.pathname === '/') {
      return '/';
    }

    const matched = menuRouteConfigs
      .filter((route) => route.path !== '/')
      .sort((a, b) => (b.path.length ?? 0) - (a.path.length ?? 0))
      .find((route) => location.pathname.startsWith(route.path));

    return matched?.path || location.pathname;
  }, [location.pathname]);

  return (
    <Sider role="navigation" aria-label={'Main Navigation'} collapsible collapsed={collapsed} onCollapse={(value) => setCollapsed(value)}>
      <Menu
        theme="dark"
        defaultSelectedKeys={['/']}
        selectedKeys={[selectedKey]}
        mode="inline"
        items={items}
        onClick={({ key }) => navigate(String(key))}
      />
    </Sider>
  );
};

export default AppSider;
