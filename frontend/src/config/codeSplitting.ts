/**
 * 代码分割配置
 * 定义哪些组件应该被懒加载和代码分割
 */

// ============ 页面级代码分割 ============

// 主要页面 - 立即加载
export const criticalPages = [
  'Dashboard',
  'Login',
  'Home'
];

// 按需加载页面
export const lazyPages = {
  // 项目管理
  Projects: () => import('../pages/Projects'),
  ProjectDetail: () => import('../pages/ProjectDetail'),

  // 代码审查
  CodeAnalysis: () => import('../pages/CodeAnalysis'),

  // 设置和配置
  Settings: () => import('../pages/Settings'),

  // 工具页面
  HelpAndAchievementsPage: () => import('../pages/HelpAndAchievementsPage'),
  Search: () => import('../pages/Search'),
  Sessions: () => import('../pages/Sessions'),
  Versions: () => import('../pages/Versions'),
  Baselines: () => import('../pages/Baselines'),
  BaselineDetailPage: () => import('../pages/BaselineDetailPage'),
  ProjectList: () => import('../pages/ProjectList'),
  ProjectArchivePage: () => import('../pages/ProjectArchivePage'),
  ProjectDangerZonePage: () => import('../pages/ProjectDangerZonePage'),
  ProjectFormPage: () => import('../pages/ProjectFormPage'),
  ProjectImportPage: () => import('../pages/ProjectImportPage'),
  QuickStartPage: () => import('../pages/QuickStartPage'),
  Register: () => import('../pages/Register'),
  ResponsiveTest: () => import('../pages/ResponsiveTest'),
  SessionDetail: () => import('../pages/SessionDetail'),
  VersionDiffViewer: () => import('../pages/VersionDiffViewer')
};

// ============ 组件级代码分割 ============

// 重型组件 - 按需加载 (仅包含实际存在的组件)
export const heavyComponents = {
  // 未来可以添加实际存在的重型组件
};

// ============ 工具函数级代码分割 ============

// 工具库 - 按需加载 (仅包含实际存在的工具)
export const utils = {
  // 未来可以添加实际存在的工具函数
};

// ============ 预加载策略 ============

/**
 * 预加载配置
 */
export const preloadConfig = {
  // 鼠标悬停预加载
  hoverPreload: {
    enabled: true,
    delay: 100, // 鼠标悬停 100ms 后预加载
    components: []
  },

  // 视口预加载
  viewportPreload: {
    enabled: true,
    rootMargin: '50px', // 距离视口 50px 时预加载
    components: []
  },

  // 路由预加载
  routePreload: {
    enabled: true,
    routes: [
      '/dashboard',
      '/projects',
      '/analysis'
    ]
  }
};

/**
 * 缓存策略
 */
export const cacheConfig = {
  // 组件缓存时间 (仅包含实际存在的组件)
  componentCache: {},

  // 工具库缓存时间 (仅包含实际存在的工具)
  utilityCache: {}
};

/**
 * 错误处理配置
 */
export const errorConfig = {
  // 重试策略
  retryStrategy: {
    maxRetries: 3,
    retryDelay: 1000, // 1 秒
    backoffMultiplier: 2
  },

  // 降级策略
  fallbackStrategy: {
    enabled: true,
    fallbackComponents: {}
  },

  // 错误报告
  errorReporting: {
    enabled: true,
    reportTo: '/api/errors',
    includeStackTrace: true
  }
};

/**
 * 性能监控配置
 */
export const performanceConfig = {
  // 加载时间监控
  loadingMetrics: {
    enabled: true,
    reportTo: '/api/metrics',
    thresholds: {
      componentLoadTime: 2000, // 2 秒
      pageLoadTime: 3000, // 3 秒
      bundleSize: 500 * 1024 // 500KB
    }
  },

  // 内存使用监控
  memoryMetrics: {
    enabled: true,
    reportInterval: 30000, // 30 秒
    thresholds: {
      heapUsed: 50 * 1024 * 1024, // 50MB
      heapTotal: 100 * 1024 * 1024 // 100MB
    }
  }
};

// ============ Webpack 代码分割配置 ============

/**
 * Webpack 魔法注释配置
 */
export const webpackMagicComments = {
  // 预加载
  preload: '/* webpackPreload: true */',

  // 预获取
  prefetch: '/* webpackPrefetch: true */',

  // 块命名
  chunkName: (name: string) => `/* webpackChunkName: "${name}" */`,

  // 模式
  mode: (mode: string) => `/* webpackMode: "${mode}" */`
};

/**
 * 动态导入包装器
 */
export const createDynamicImport = (
  importFunc: () => Promise<any>,
  chunkName?: string,
  preload?: boolean,
  prefetch?: boolean
) => {
  let magicComment = '';

  if (chunkName) {
    magicComment += webpackMagicComments.chunkName(chunkName);
  }

  if (preload) {
    magicComment += webpackMagicComments.preload;
  }

  if (prefetch) {
    magicComment += webpackMagicComments.prefetch;
  }

  // 注意：实际使用时需要手动添加魔法注释
  return importFunc;
};

// ============ 使用示例 ============

/*
// 1. 页面级代码分割
import { lazyPages } from '../config/codeSplitting';
import { createLazyRoute } from '../utils/virtualization';

const routes = [
  {
    path: '/dashboard',
    component: createLazyRoute(lazyPages.Dashboard)
  },
  {
    path: '/projects',
    component: createLazyRoute(lazyPages.Projects)
  }
];

// 2. 组件级代码分割
import { heavyComponents } from '../config/codeSplitting';
import { lazyLoadComponent } from '../utils/virtualization';

const LazyChart = lazyLoadComponent(heavyComponents.Charts.LineChart);

// 3. 工具库代码分割
import { utilityLibraries } from '../config/codeSplitting';

const loadCsvParser = utilityLibraries.dataProcessing.csvParser;

// 4. 预加载策略
import { preloadConfig } from '../config/codeSplitting';

if (preloadConfig.routePreload.enabled) {
  // 预加载关键路由
  preloadConfig.routePreload.routes.forEach(route => {
    // 实现预加载逻辑
  });
}
*/