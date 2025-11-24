/* eslint-disable prettier/prettier */
import React, { ComponentType, ReactElement } from 'react';
import {
  DashboardOutlined,
  TrophyOutlined,
  CodeOutlined,
  SettingOutlined,
  GithubOutlined,
  ProjectOutlined,
  HistoryOutlined,
  BranchesOutlined,
  SearchOutlined,
  DatabaseOutlined,
  QuestionCircleOutlined,
  MonitorOutlined,
} from '@ant-design/icons';

import Dashboard from '../pages/Dashboard';
import QuickStartPage from '../pages/QuickStartPage';
import CodeAnalysis from '../pages/CodeAnalysis';
import GitHubConnect from '../components/GitHubConnect';
import Projects from '../pages/Projects';
import ProjectList from '../pages/ProjectList';
import ProjectDetail from '../pages/ProjectDetail';
import Sessions from '../pages/Sessions';
import SessionDetail from '../pages/SessionDetail';
import Versions from '../pages/Versions';
import VersionDiffViewer from '../pages/VersionDiffViewer';
import Search from '../pages/Search';
import Baselines from '../pages/Baselines';
import BaselineDetailPage from '../pages/BaselineDetailPage';
import Settings from '../pages/Settings';
import HelpAndAchievementsPage from '../pages/HelpAndAchievementsPage';
import Login from '../pages/Login';
import Register from '../pages/Register';
import NotFound from '../pages/NotFound';
import ProjectFormPage from '../pages/ProjectFormPage';
import ProjectDangerZonePage from '../pages/ProjectDangerZonePage';
import ProjectImportPage from '../pages/ProjectImportPage';
import ProjectArchivePage from '../pages/ProjectArchivePage';
import ResponsiveTest from '../pages/ResponsiveTest';
import Monitoring from '../pages/Monitoring';
import Profile from '../pages/Profile';

export interface AppRouteConfig {
  path: string;
  component: ComponentType;
  requiresAuth?: boolean;
  showInMenu?: boolean;
  labelKey?: string;
  fallbackLabel?: string;
  icon?: ReactElement;
  menuOrder?: number;
}

export const appRoutes: AppRouteConfig[] = [
  {
    path: '/',
    component: Dashboard,
    requiresAuth: true,
    showInMenu: true,
    labelKey: 'menu.dashboard',
    fallbackLabel: 'Dashboard',
    icon: <DashboardOutlined />,
    menuOrder: 0
  },
  {
    path: '/quick-start',
    component: QuickStartPage,
    requiresAuth: true,
    showInMenu: true,
    labelKey: 'menu.quick_start',
    fallbackLabel: 'Quick Start',
    icon: <TrophyOutlined />,
    menuOrder: 1
  },
  {
    path: '/analysis',
    component: CodeAnalysis,
    requiresAuth: true,
    showInMenu: true,
    labelKey: 'menu.code_analysis',
    fallbackLabel: 'Code Analysis',
    icon: <CodeOutlined />,
    menuOrder: 2
  },

  {
    path: '/github-connect',
    component: GitHubConnect,
    requiresAuth: true,
    showInMenu: true,
    labelKey: 'menu.github',
    fallbackLabel: 'GitHub Connect',
    icon: <GithubOutlined />,
    menuOrder: 4
  },
  {
    path: '/projects',
    component: Projects,
    requiresAuth: true,
    showInMenu: true,
    labelKey: 'menu.projects',
    fallbackLabel: 'Projects',
    icon: <ProjectOutlined />,
    menuOrder: 5
  },
  {
    path: '/projects/list',
    component: ProjectList,
    requiresAuth: true,
    showInMenu: false
  },
  {
    path: '/projects/:id',
    component: ProjectDetail,
    requiresAuth: true,
    showInMenu: false
  },
  {
    path: '/projects/new',
    component: ProjectFormPage,
    requiresAuth: true,
    showInMenu: false
  },
  {
    path: '/projects/:id/edit',
    component: ProjectFormPage,
    requiresAuth: true,
    showInMenu: false
  },
  {
    path: '/projects/:id/danger-zone',
    component: ProjectDangerZonePage,
    requiresAuth: true,
    showInMenu: false
  },
  {
    path: '/projects/import',
    component: ProjectImportPage,
    requiresAuth: true,
    showInMenu: false
  },
  {
    path: '/projects/archive',
    component: ProjectArchivePage,
    requiresAuth: true,
    showInMenu: false
  },
  {
    path: '/sessions',
    component: Sessions,
    requiresAuth: true,
    showInMenu: true,
    labelKey: 'menu.sessions',
    fallbackLabel: 'Sessions',
    icon: <HistoryOutlined />,
    menuOrder: 6
  },
  {
    path: '/sessions/:id',
    component: SessionDetail,
    requiresAuth: true,
    showInMenu: false
  },
  {
    path: '/versions',
    component: Versions,
    requiresAuth: true,
    showInMenu: true,
    labelKey: 'menu.versions',
    fallbackLabel: 'Versions',
    icon: <BranchesOutlined />,
    menuOrder: 7
  },
  {
    path: '/versions/diff',
    component: VersionDiffViewer,
    requiresAuth: true,
    showInMenu: false
  },
  {
    path: '/search',
    component: Search,
    requiresAuth: true,
    showInMenu: true,
    labelKey: 'menu.search',
    fallbackLabel: 'Search',
    icon: <SearchOutlined />,
    menuOrder: 8
  },
  {
    path: '/baselines',
    component: Baselines,
    requiresAuth: true,
    showInMenu: true,
    labelKey: 'menu.baselines',
    fallbackLabel: 'Baselines',
    icon: <DatabaseOutlined />,
    menuOrder: 9
  },
  {
    path: '/baselines/:id',
    component: BaselineDetailPage,
    requiresAuth: true,
    showInMenu: false
  },


  {
    path: '/settings',
    component: Settings,
    requiresAuth: true,
    showInMenu: true,
    labelKey: 'menu.settings',
    fallbackLabel: 'Settings',
    icon: <SettingOutlined />,
    menuOrder: 13
  },
  {
    path: '/help-achievements',
    component: HelpAndAchievementsPage,
    requiresAuth: true,
    showInMenu: true,
    labelKey: 'menu.help_achievements',
    fallbackLabel: 'Help & Achievements',
    icon: <QuestionCircleOutlined />,
    menuOrder: 14
  },
  {
    path: '/monitoring',
    component: Monitoring,
    requiresAuth: true,
    showInMenu: true,
    labelKey: 'menu.monitoring',
    fallbackLabel: 'Monitoring',
    icon: <MonitorOutlined />,
    menuOrder: 10
  },
  {
    path: '/profile',
    component: Profile,
    requiresAuth: true,
    showInMenu: false,
  },
  {
    path: '/responsive-test',
    component: ResponsiveTest,
    requiresAuth: false,
    showInMenu: false,
  },
  {
    path: '/login',
    component: Login,
    requiresAuth: false,
    showInMenu: false,
  },
  {
    path: '/register',
    component: Register,
    requiresAuth: false,
    showInMenu: false,
  },
  {
    path: '*',
    component: NotFound,
    requiresAuth: false,
    showInMenu: false,
  }
];

export const menuRouteConfigs = appRoutes
  .filter((route) => route.showInMenu)
  .sort((a, b) => (a.menuOrder ?? 0) - (b.menuOrder ?? 0));

